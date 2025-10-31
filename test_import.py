# Quick import test (optional): run with your Python env to validate imports without ComfyUI.
from pathlib import Path
import sys

# Ensure repo root on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import comfyui_emu35_node as node
    print("Imported comfyui_emu35_node successfully.")
    mr = node._get_comfy_models_root()
    print(f"Detected models root: {mr}")
except Exception as e:
    print(f"Import failed: {e}")
