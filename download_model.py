from huggingface_hub import snapshot_download
import os
DIRPATH = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(DIRPATH, "huggingface/Qwen3-0.6B")
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir=model_path,
    local_dir_use_symlinks=False,
    resume_download=True
)