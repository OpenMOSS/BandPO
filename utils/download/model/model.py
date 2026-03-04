from huggingface_hub import snapshot_download
snapshot_download(
#   repo_id="Yuan-Li-FNLP/R3-RAG-Qwen",
#   repo_id="Yuan-Li-FNLP/R3-RAG-CS-Qwen",
#   repo_id="Yuan-Li-FNLP/R3-RAG-Llama",
  repo_id="Yuan-Li-FNLP/R3-RAG-CS-Llama",
#   local_dir="/remote-home1/yli/Workspace/R3-RAG/ExperimentData/Models/qwen/RLHF",
#   local_dir="/remote-home1/yli/Workspace/R3-RAG/ExperimentData/Models/qwen/SFT",
#   local_dir="/remote-home1/yli/Workspace/R3-RAG/ExperimentData/Models/llama/RLHF",
  local_dir="/remote-home1/yli/Workspace/R3-RAG/ExperimentData/Models/llama/SFT",
  proxies={"https": "http://localhost:7890"},
  max_workers=8
)