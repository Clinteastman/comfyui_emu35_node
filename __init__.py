"""
ComfyUI custom node for Emu3.5 (Text-to-Image first).

Features
- Auto-downloads models from Hugging Face into ComfyUI/models.
- CUDA fp16 by default; device default "cuda:0" (out-of-range CUDA indices resolve to "cuda:0").
- Optional offload via device_map="auto" toggle.
- Sequential batching with base_seed + index.

Notes
- This node is developed inside the Emu3.5 repository; it imports from the local ./src tree.
- When moving to ComfyUI/custom_nodes later, you can either vendor the minimal src modules
  or package Emu3.5 as an installable dependency.
"""

from __future__ import annotations

import os
import sys
import math
import types
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import torch

# Try to import ComfyUI folder paths utilities if available (when running inside ComfyUI)
_FOLDER_PATHS = None
try:
    import folder_paths as _FOLDER_PATHS  # type: ignore
except Exception:
    _FOLDER_PATHS = None


# Ensure repo root is on sys.path so we can import from ./src
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Local imports from this repo
from src.utils.model_utils import build_emu3p5
from src.emu3p5 import Emu3ForCausalLM, Emu3Config
from src.utils.generation_utils import generate, multimodal_decode
from src.utils.input_utils import build_image
from src.tokenizer_emu3_ibq.tokenization_emu3 import Emu3Tokenizer

# For optional reference-image support in future
from PIL import Image


# ---------------------------
# Helpers for models location
# ---------------------------

HF_VQ_REPO = "BAAI/Emu3.5-VisionTokenizer"
HF_DEFAULT_MODEL = "BAAI/Emu3.5"           # default choice
HF_ALT_IMAGE_MODEL = "BAAI/Emu3.5-Image"   # user-selectable


def _get_comfy_models_root() -> Path:
    """Best-effort detection of the ComfyUI models root directory.

    Returns a writable path. Falls back to ./models under repo if not running in ComfyUI.
    """
    # 1) Environment override
    env_override = os.environ.get("COMFYUI_MODELS_DIR")
    if env_override:
        return Path(env_override).resolve()

    # 2) Use ComfyUI's folder_paths if available
    if _FOLDER_PATHS is not None:
        try:
            # "checkpoints" is typically models/checkpoints; take the parent
            ckpts = _FOLDER_PATHS.get_folder_paths("checkpoints")
            if ckpts and len(ckpts) > 0:
                return Path(ckpts[0]).resolve().parents[1]  # .../ComfyUI/models
        except Exception:
            pass

    # 3) Fallback to repo-local models folder
    return (_REPO_ROOT / "models").resolve()


def _ensure_hf_repo(repo_id: str, dest_dir: Path) -> Path:
    """Download (if missing) a Hugging Face repo snapshot into dest_dir/repo_name.

    Uses huggingface_hub.snapshot_download with Windows-friendly symlink settings.
    """
    local_repo_dir = dest_dir / Path(repo_id).name
    # Allow explicit overrides
    if repo_id == HF_VQ_REPO:
        override = os.environ.get("EMU35_VQ_DIR")
        if override:
            return Path(override).resolve()
    else:
        override = os.environ.get("EMU35_MODEL_DIR")
        if override:
            return Path(override).resolve()

    # If already present (contains some known files), skip download
    if local_repo_dir.exists() and any(local_repo_dir.iterdir()):
        return local_repo_dir

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "Missing dependency huggingface_hub. Please install it (pip install huggingface_hub) "
            "or add it to your ComfyUI environment."
        ) from e

    local_repo_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_repo_dir),
        local_dir_use_symlinks=False,  # Windows-friendly
        revision=None,
        allow_patterns=None,
        ignore_patterns=None,
        tqdm_class=None,
    )
    return local_repo_dir


def _build_prompts(prompt: str, with_image: bool = False) -> Tuple[str, str]:
    """Replicates the template construction from configs/config.py for T2I."""
    task_str = "t2i"
    if with_image:
        unc_p = "<|extra_203|>You are a helpful assistant. USER: <|IMAGE|> ASSISTANT: <|extra_100|>"
        tmpl = f"<|extra_203|>You are a helpful assistant for {task_str} task. USER: {{question}}<|IMAGE|> ASSISTANT: <|extra_100|>"
    else:
        unc_p = "<|extra_203|>You are a helpful assistant. USER:  ASSISTANT: <|extra_100|>"
        tmpl = f"<|extra_203|>You are a helpful assistant for {task_str} task. USER: {{question}} ASSISTANT: <|extra_100|>"
    return unc_p, tmpl.format(question=prompt)


def _build_cfg(
    guidance: float,
    unconditional_type: str,
    image_cfg_scale: float,
    text_top_k: int,
    text_top_p: float,
    text_temperature: float,
    image_top_k: int,
    image_top_p: float,
    image_temperature: float,
    max_new_tokens: int,
    use_differential_sampling: bool,
) -> types.SimpleNamespace:
    sampling_params: Dict[str, Any] = dict(
        use_cache=True,
        # text sampling
        text_top_k=int(text_top_k),
        text_top_p=float(text_top_p),
        text_temperature=float(text_temperature),
        # image sampling
        image_top_k=int(image_top_k),
        image_top_p=float(image_top_p),
        image_temperature=float(image_temperature),
        # general
        top_k=131072,
        top_p=1.0,
        temperature=1.0,
        num_beams_per_group=1,
        num_beam_groups=1,
        diversity_penalty=0.0,
        max_new_tokens=int(max_new_tokens),
        guidance_scale=1.0,
        use_differential_sampling=bool(use_differential_sampling),
    )
    sampling_params["do_sample"] = sampling_params["num_beam_groups"] <= 1
    sampling_params["num_beams"] = (
        sampling_params["num_beams_per_group"] * sampling_params["num_beam_groups"]
    )

    cfg = types.SimpleNamespace(
        streaming=False,
        unconditional_type=unconditional_type,
        classifier_free_guidance=float(guidance),
        sampling_params=sampling_params,
        image_cfg_scale=float(image_cfg_scale),
    )
    # special tokens map filled after tokenizer is built
    return cfg


def _comfy_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]
    image_tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)
    arr = (image_tensor.numpy() * 255.0)
    import numpy as np
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr.round().astype("uint8")
    return Image.fromarray(arr)


# ---------------------------
# Node: Model Loader
# ---------------------------

_MODEL_CACHE: Dict[Tuple[str, str, str, str, bool], Tuple[Any, Any, Any, Dict[str, int]]] = {}


class Emu35_Load:
    CATEGORY = "emu3.5"
    RETURN_TYPES = ("EMU35_MODEL",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo": ("STRING", {"default": HF_DEFAULT_MODEL}),
                "precision": ("STRING", {"default": "fp16", "choices": ["fp16", "bf16"]}),
                "device": ("STRING", {"default": "cuda:0"}),
                "offload": ("BOOLEAN", {"default": False, "label": "device_map='auto'"}),
                "attn_backend": ("STRING", {"default": "auto", "choices": ["auto", "flash_attn", "sdpa"]}),
            }
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda:"):
            try:
                idx = int(device.split(":")[1])
                if torch.cuda.is_available() and torch.cuda.device_count() > idx:
                    return device
            except Exception:
                pass
            # fallback
            if torch.cuda.is_available():
                return "cuda:0"
        return device

    def load(self, model_repo: str, precision: str, device: str, offload: bool, attn_backend: str):
        device_resolved = self._resolve_device(device)

        models_root = _get_comfy_models_root() / "Emu3.5"
        models_root.mkdir(parents=True, exist_ok=True)

        # Ensure repos exist locally
        model_dir = _ensure_hf_repo(model_repo, models_root)
        vq_dir = _ensure_hf_repo(HF_VQ_REPO, models_root)

        # Cache key
        key = (str(model_dir), device_resolved, precision, attn_backend, bool(offload))
        if key in _MODEL_CACHE:
            return (_MODEL_CACHE[key],)

        # Determine dtype
        torch_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

        # Attention backend selection for stability on Windows
        attn_choice = attn_backend
        if attn_choice == "auto":
            # Prefer flash_attn, fall back to sdpa
            try:
                import flash_attn  # type: ignore
                attn_choice = "flash_attn"
            except Exception:
                attn_choice = "sdpa"

        # Build model/tokenizer/vq
        model_device = "auto" if offload else device_resolved
        try:
            model, tokenizer, vq_model = build_emu3p5(
                str(model_dir),
                str(_REPO_ROOT / "src" / "tokenizer_emu3_ibq"),
                str(vq_dir),
                vq_type="ibq",
                model_device=model_device,
                vq_device=device_resolved if device_resolved.startswith("cuda") else "cpu",
            )
        except Exception as e:
            # Fallback: manual load with SDPA attention if flash_attn path failed
            if attn_choice == "sdpa":
                try:
                    model_config = Emu3Config.from_pretrained(str(model_dir), trust_remote_code=True)
                    model = Emu3ForCausalLM.from_pretrained(
                        str(model_dir),
                        config=model_config,
                        torch_dtype=torch_dtype,
                        device_map=model_device,
                        attn_implementation="sdpa",
                        trust_remote_code=True,
                    )
                    model.eval()
                    # Load tokenizer and vq tokenizer akin to build_emu3p5
                    tokenizer = Emu3Tokenizer.from_pretrained(
                        str(_REPO_ROOT / "src" / "tokenizer_emu3_ibq"),
                        special_tokens_file=str(_REPO_ROOT / "src" / "tokenizer_emu3_ibq" / "emu3_vision_tokens.txt"),
                        trust_remote_code=True,
                    )
                    # Set special tokens as build_emu3p5 does
                    tokenizer.bos_token = "<|extra_203|>"
                    tokenizer.eos_token = "<|extra_204|>"
                    tokenizer.pad_token = "<|endoftext|>"
                    tokenizer.eol_token = "<|extra_200|>"
                    tokenizer.eof_token = "<|extra_201|>"
                    tokenizer.tms_token = "<|extra_202|>"
                    tokenizer.img_token = "<|image token|>"
                    tokenizer.boi_token = "<|image start|>"
                    tokenizer.eoi_token = "<|image end|>"
                    tokenizer.bss_token = "<|extra_100|>"
                    tokenizer.ess_token = "<|extra_101|>"
                    tokenizer.bog_token = "<|extra_60|>"
                    tokenizer.eog_token = "<|extra_61|>"
                    tokenizer.boc_token = "<|extra_50|>"
                    tokenizer.eoc_token = "<|extra_51|>"

                    # Vision tokenizer via build_vision_tokenizer
                    from src.vision_tokenizer import build_vision_tokenizer as _bvt
                    vq_model = _bvt("ibq", str(vq_dir), device=(device_resolved if device_resolved.startswith("cuda") else "cpu"))
                except Exception as e2:
                    raise RuntimeError(f"Failed to load Emu3.5 model with SDPA fallback: {e2}") from e
            else:
                raise RuntimeError(f"Failed to load Emu3.5 model from {model_dir}: {e}") from e

        # Adjust dtype/attention after load when build_emu3p5 doesn't expose knobs
        try:
            if torch_dtype is not None:
                model.to(dtype=torch_dtype)
        except Exception:
            pass

        # Try to set attention impl
        try:
            if attn_choice == "sdpa" and hasattr(model, "set_attn_implementation"):
                model.set_attn_implementation("sdpa")
        except Exception:
            pass

        # Special token ids
        from configs import config as base_cfg  # only for token strings
        special_tokens_map: Dict[str, int] = {}
        for k, v in base_cfg.special_tokens.items():
            special_tokens_map[k] = tokenizer.encode(v)[0]

        handle = (model, tokenizer, vq_model, special_tokens_map)
        _MODEL_CACHE[key] = handle
        return (handle,)


# ---------------------------
# Node: Text-to-Image (batch via loop)
# ---------------------------

class Emu35_T2I:
    CATEGORY = "emu3.5"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emu35_model": ("EMU35_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A clay astronaut on Mars."}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 32}),
                "base_seed": ("INT", {"default": 123456789, "min": 0, "max": 2**31-1}),
                # Guidance and sampling
                "guidance": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "unconditional_type": ("STRING", {"default": "no_text", "choices": ["no_text", "no_text_img_cfg"]}),
                "image_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 32768, "min": 1, "max": 262144}),
                # differential sampling
                "use_differential_sampling": ("BOOLEAN", {"default": True}),
                "text_top_k": ("INT", {"default": 1024, "min": 0, "max": 200000}),
                "text_top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "image_top_k": ("INT", {"default": 10240, "min": 0, "max": 200000}),
                "image_top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
            },
        }

    def _seed_all(self, seed: int):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run(
        self,
        emu35_model: Tuple[Any, Any, Any, Dict[str, int]],
        prompt: str,
        num_images: int,
        base_seed: int,
        guidance: float,
        unconditional_type: str,
        image_cfg_scale: float,
        max_new_tokens: int,
        use_differential_sampling: bool,
        text_top_k: int,
        text_top_p: float,
        text_temperature: float,
        image_top_k: int,
        image_top_p: float,
        image_temperature: float,
    ):
        model, tokenizer, vq_model, special_tokens_map = emu35_model

        # Build cfg and prompts
        cfg = _build_cfg(
            guidance=guidance,
            unconditional_type=unconditional_type,
            image_cfg_scale=image_cfg_scale,
            text_top_k=text_top_k,
            text_top_p=text_top_p,
            text_temperature=text_temperature,
            image_top_k=image_top_k,
            image_top_p=image_top_p,
            image_temperature=image_temperature,
            max_new_tokens=max_new_tokens,
            use_differential_sampling=use_differential_sampling,
        )
        cfg.special_token_ids = special_tokens_map

        unc_prompt, cond_prompt = _build_prompts(prompt, with_image=False)

        images: List[torch.Tensor] = []
        device = model.device if hasattr(model, "device") else (next(model.parameters()).device)

        for i in range(int(num_images)):
            self._seed_all(int(base_seed) + i)

            input_ids = tokenizer.encode(cond_prompt, return_tensors="pt", add_special_tokens=False).to(device)
            if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
                BOS = torch.Tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
                input_ids = torch.cat([BOS, input_ids], dim=1)

            unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(device)
            full_unc_ids = None  # Optional: could expose img_unc in future

            # Run generation (non-streaming)
            for result in generate(cfg, model, tokenizer, input_ids, unconditional_ids, full_unc_ids):
                result_str = tokenizer.decode(result, skip_special_tokens=False)
                mm_out = multimodal_decode(result_str, tokenizer, vq_model)
                # pick first image
                img_pil = None
                for (typ, payload) in mm_out:
                    if typ == "image":
                        img_pil = payload
                        break
                if img_pil is None:
                    # If no image in output, create a black placeholder to avoid breaking the graph
                    img_pil = Image.new("RGB", (512, 512), color=(0, 0, 0))

                # PIL -> Comfy IMAGE tensor [B,H,W,C], float32 [0,1]
                import numpy as np
                arr = np.array(img_pil).astype("float32") / 255.0
                t = torch.from_numpy(arr)[None, ...]  # [1,H,W,3]
                images.append(t)

        # Stack along batch
        out = torch.cat(images, dim=0) if len(images) > 1 else images[0]
        return (out,)


class Emu35_I2I:
    CATEGORY = "emu3.5"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emu35_model": ("EMU35_MODEL",),
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Refine the reference scene with extra clay astronaut details."}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 32}),
                "base_seed": ("INT", {"default": 123456789, "min": 0, "max": 2**31 - 1}),
                "guidance": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "unconditional_type": ("STRING", {"default": "no_text_img_cfg", "choices": ["no_text", "no_text_img_cfg"]}),
                "image_cfg_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 32768, "min": 1, "max": 262144}),
                "use_differential_sampling": ("BOOLEAN", {"default": True}),
                "text_top_k": ("INT", {"default": 1024, "min": 0, "max": 200000}),
                "text_top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "image_top_k": ("INT", {"default": 10240, "min": 0, "max": 200000}),
                "image_top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_temperature": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01}),
                "image_area": ("INT", {"default": 518400, "min": 65536, "max": 1048576, "step": 256}),
            },
        }

    def run(
        self,
        emu35_model: Tuple[Any, Any, Any, Dict[str, int]],
        reference_image: torch.Tensor,
        prompt: str,
        num_images: int,
        base_seed: int,
        guidance: float,
        unconditional_type: str,
        image_cfg_scale: float,
        max_new_tokens: int,
        use_differential_sampling: bool,
        text_top_k: int,
        text_top_p: float,
        text_temperature: float,
        image_top_k: int,
        image_top_p: float,
        image_temperature: float,
        image_area: int,
    ):
        model, tokenizer, vq_model, special_tokens_map = emu35_model

        cfg = _build_cfg(
            guidance=guidance,
            unconditional_type=unconditional_type,
            image_cfg_scale=image_cfg_scale,
            text_top_k=text_top_k,
            text_top_p=text_top_p,
            text_temperature=text_temperature,
            image_top_k=image_top_k,
            image_top_p=image_top_p,
            image_temperature=image_temperature,
            max_new_tokens=max_new_tokens,
            use_differential_sampling=use_differential_sampling,
        )
        cfg.special_token_ids = special_tokens_map
        cfg.image_area = int(image_area)

        unc_template, cond_template = _build_prompts(prompt, with_image=True)
        ref_pil = _comfy_image_to_pil(reference_image)
        image_token_str = build_image(ref_pil, cfg, tokenizer, vq_model)
        cond_prompt = cond_template.replace("<|IMAGE|>", image_token_str)
        unc_prompt = unc_template.replace("<|IMAGE|>", image_token_str)

        images: List[torch.Tensor] = []
        device = model.device if hasattr(model, "device") else (next(model.parameters()).device)

        for i in range(int(num_images)):
            Emu35_T2I._seed_all(self, int(base_seed) + i)

            input_ids = tokenizer.encode(cond_prompt, return_tensors="pt", add_special_tokens=False).to(device)
            if input_ids[0, 0] != cfg.special_token_ids["BOS"]:
                BOS = torch.Tensor([[cfg.special_token_ids["BOS"]]], device=input_ids.device, dtype=input_ids.dtype)
                input_ids = torch.cat([BOS, input_ids], dim=1)

            unconditional_ids = tokenizer.encode(unc_prompt, return_tensors="pt", add_special_tokens=False).to(device)
            full_unc_ids = None

            for result in generate(cfg, model, tokenizer, input_ids, unconditional_ids, full_unc_ids):
                result_str = tokenizer.decode(result, skip_special_tokens=False)
                mm_out = multimodal_decode(result_str, tokenizer, vq_model)
                img_pil = None
                for (typ, payload) in mm_out:
                    if typ == "image":
                        img_pil = payload
                        break
                if img_pil is None:
                    img_pil = Image.new("RGB", ref_pil.size, color=(0, 0, 0))

                import numpy as np

                arr = np.array(img_pil).astype("float32") / 255.0
                t = torch.from_numpy(arr)[None, ...]
                images.append(t)

        out = torch.cat(images, dim=0) if len(images) > 1 else images[0]
        return (out,)


NODE_CLASS_MAPPINGS = {
    "Emu3.5 Load (fp16)": Emu35_Load,
    "Emu3.5 T2I (Batch)": Emu35_T2I,
    "Emu3.5 I2I (Batch)": Emu35_I2I,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Emu3.5 Load (fp16)": "Emu3.5 Load (fp16)",
    "Emu3.5 T2I (Batch)": "Emu3.5 T2I (Batch)",
    "Emu3.5 I2I (Batch)": "Emu3.5 I2I (Batch)",
}
