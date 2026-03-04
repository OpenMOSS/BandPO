export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download "Qwen/Qwen2.5-Math-7B-Instruct" --local-dir /remote-home1/yli/Workspace/BandPO/data/models/qwen25math/instruct/Qwen2.5-Math-7B-Instruct --local-dir-use-symlinks False

huggingface-cli download --resume-download "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --local-dir "$BandPODir/data/models/deepseek/R1/DeepSeek-R1-Distill-Qwen-1.5B" --local-dir-use-symlinks False

huggingface-cli download --resume-download "BAAI/bge-m3" --local-dir "$BandPODir/data/models/BAAI/bge-m3" --local-dir-use-symlinks False
huggingface-cli download --resume-download "1-800-BAD-CODE/sentence_boundary_detection_multilang" --local-dir "$BandPODir/data/models/1-800-BAD-CODE/sentence_boundary_detection_multilang" --local-dir-use-symlinks False

huggingface-cli download --resume-download "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --local-dir "$BandPODir/data/models/deepseek/R1/DeepSeek-R1-Distill-Qwen-7B" --local-dir-use-symlinks False

huggingface-cli download --resume-download "Qwen/Qwen2.5-3B-Instruct" --local-dir "$BandPODir/data/models/Qwen/Qwen2.5-3B-Instruct" --local-dir-use-symlinks False

huggingface-cli download --resume-download "BAAI/bge-m3" --local-dir "/remote-home1/share/models/BAAI/bge-m3" --local-dir-use-symlinks False