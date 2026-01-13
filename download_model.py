from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="./huggingface/Qwen3-0.6B/",
    local_dir_use_symlinks=False,
    resume_download=True
)