import os
from huggingface_hub import snapshot_download

model_name = "Qwen/Qwen3-4B-Base"

CACHE = os.path.expanduser(os.environ.get("CACHE", "~/.cache"))
assert CACHE, "$CACHE is empty"

hf_home = os.path.join(CACHE, "hf_home")
hf_hub_cache = os.path.join(hf_home, "hub")
tfm_cache = os.path.join(hf_home, "transformers")
os.makedirs(hf_hub_cache, exist_ok=True)
os.makedirs(tfm_cache, exist_ok=True)

os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_CACHE"] = hf_hub_cache
os.environ["TRANSFORMERS_CACHE"] = tfm_cache

local_path = os.path.join(CACHE, f"hf_models/{model_name}")
os.makedirs(local_path, exist_ok=True)

print(f"Downloading repo {model_name} directly under $CACHE ...")

snapshot_download(
    repo_id=model_name,
    local_dir=local_path,
    local_dir_use_symlinks=False,  
    cache_dir=hf_hub_cache         
)

print(f"All files saved to: {local_path}")
print("Done (CPU-only, no model loaded).")
