# Emu3.5 ComfyUI Node (in-repo)

Text-to-Image ComfyUI node for BAAI Emu3.5, developed inside this repository first. Move the `comfyui_emu35_node/` folder to `ComfyUI/custom_nodes/` later to use in ComfyUI.

## Features

- Auto-download Hugging Face repos to ComfyUI models folder:
  - `BAAI/Emu3.5` (default) or `BAAI/Emu3.5-Image`
  - `BAAI/Emu3.5-VisionTokenizer`
- CUDA fp16 by default; default device `cuda:9` with fallback to `cuda:0`.
- Optional offload via `device_map="auto"` toggle.
- Sequential batching with `base_seed + index`.

## Default behavior

- Model: `BAAI/Emu3.5` by default (you can switch to `BAAI/Emu3.5-Image`).
- Device: `cuda:9` by default. If that GPU index isn’t available, the loader falls back to `cuda:0` automatically.
- Precision: fp16 by default (bf16 optional). If flash-attn isn’t available on Windows, the node falls back to SDPA attention automatically.
- Offload: visible toggle that sets `device_map="auto"` for HF accelerate offload when VRAM is tight.

## Install deps (Windows PowerShell)

```powershell
# Inside your ComfyUI environment
pip install -r .\requirements.txt  # from the repo root (Emu3.5 requirements)
pip install -r .\comfyui_emu35_node\requirements.txt  # adds huggingface_hub
```

## Move into ComfyUI/custom_nodes

```powershell
# Copy this folder to your ComfyUI custom_nodes
# Replace the path with your ComfyUI install
Copy-Item -Recurse -Force .\comfyui_emu35_node "C:\path\to\ComfyUI\custom_nodes\comfyui_emu35_node"
```

## Usage

1. Copy or symlink `comfyui_emu35_node/` into your `%COMFYUI_ROOT%/custom_nodes/`.
2. Launch ComfyUI. Look under the "emu3.5" category for:
   - "Emu3.5 Load (fp16)"
   - "Emu3.5 T2I (Batch)"
3. On first run, the node will download the selected model(s) into `%COMFYUI_ROOT%/models/Emu3.5/`.

### Quick start workflow

You can import a ready-made workflow graph:

```powershell
# In ComfyUI, use "Load" and select this file from the repo
%CD%\comfyui_emu35_node\workflows\emu35_t2i_example.json
```

That workflow:

- Loads `BAAI/Emu3.5` on `cuda:9` (fallback `cuda:0`), fp16; offload disabled by default
- Runs “Emu3.5 T2I (Batch)” with a sample prompt, batch=1, guidance=2.0, and default sampling knobs
- Saves the image using the standard `SaveImage` node

## Node parameters

- Emu3.5 Load (fp16)
  - `model_repo`: string; default `BAAI/Emu3.5` (you can switch to `BAAI/Emu3.5-Image`).
  - `precision`: `fp16` (default) or `bf16`.
  - `device`: string; default `cuda:9` (falls back to `cuda:0` if index 9 is missing).
  - `offload`: boolean; when true, uses `device_map="auto"` offload.
  - `attn_backend`: `auto` (default), `flash_attn`, or `sdpa`. `auto` tries flash-attn, falls back to SDPA.

- Emu3.5 T2I (Batch)
  - `prompt`: text prompt (multiline supported).
  - `num_images`: batch size; runs sequentially.
  - `base_seed`: base seed; image i uses `base_seed + i` for reproducible variety.
  - `guidance`: classifier-free guidance scale (default 2.0; consider ~5.0 for `Emu3.5-Image`).
  - `unconditional_type`: `no_text` (default) or `no_text_img_cfg`.
  - `image_cfg_scale`: extra image CFG weight when `no_text_img_cfg` is used.
  - `max_new_tokens`: cap on generated tokens; default 32768.
  - Differential sampling knobs (defaults mirror the repo):
    - Text: `text_top_k`, `text_top_p`, `text_temperature`
    - Image: `image_top_k`, `image_top_p`, `image_temperature`
  - Not applicable: negative prompts, steps/samplers (this is an AR model, not diffusion).

## Environment overrides

- `COMFYUI_MODELS_DIR`: Override ComfyUI models root detection.
- `EMU35_MODEL_DIR`: Full path to a local Emu3.5 model directory.
- `EMU35_VQ_DIR`: Full path to a local Emu3.5-VisionTokenizer directory.

## Notes

- Negative prompt strings and "steps/sampler" are not applicable to this AR pipeline.
- T2I resolution is decided by the model; no explicit width/height inputs.
- VRAM for 34B models is large; consider enabling offload (`device_map=auto`) if you run out of memory.
- If `flash-attn` is unavailable on Windows, the node will fall back to SDPA attention.

## Models and cache locations

- If running under ComfyUI, weights download into `%COMFYUI_ROOT%/models/Emu3.5/`.
  - `.../Emu3.5/Emu3.5` or `.../Emu3.5/Emu3.5-Image`
  - `.../Emu3.5/Emu3.5-VisionTokenizer`
- If not in ComfyUI, the node falls back to a local `./models/` folder in this repo.
- You can point to existing local folders using the environment overrides above.

## Troubleshooting

- Missing dependency: if `huggingface_hub` is not installed, run the install commands in “Install deps”.
- Flash-attn issues on Windows: select `attn_backend=sdpa` in the loader or keep `auto` (it falls back to SDPA automatically).
- Device not found: if `cuda:9` isn’t available, the loader will use `cuda:0`. You can set `device` explicitly.
- OOM on large GPUs: enable `offload` (device_map=auto), reduce `num_images` to 1, reduce `max_new_tokens`, or use the base `Emu3.5` model instead of `Emu3.5-Image`.

## Roadmap (optional)

- Image-to-Image support using the repo’s VQ encode path.
- Secondary text outputs (e.g., global_cot/image_cot) as optional outputs.
