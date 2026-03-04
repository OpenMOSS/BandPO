#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModel

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_bge_m3(device):
    model_name = "/remote-home1/yli/Workspace/BandPO/data/models/BAAI/bge-m3"
    print(f"Loading model: {model_name} on device: {device}")

    # A800 建议直接用 bfloat16，显存+速度都更友好
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,   # 如果你的 torch 不支持 bf16，可以改为 torch.float16 / float32
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer, model

@torch.no_grad()
def encode_texts(tokenizer, model, texts, device, batch_size=4, max_length=512):
    """
    用 BGE-M3 生成 dense embedding。
    按官方说明，BGE 族模型使用 [CLS] 的 last hidden state 作为句向量。:contentReference[oaicite:3]{index=3}
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # 官方 FAQ 里建议 lower().strip()，保证和训练时一致的分词行为:contentReference[oaicite:4]{index=4}
        norm_texts = [t.lower().strip() for t in batch_texts]

        inputs = tokenizer(
            norm_texts,
            padding=True,
            truncation=True,
            max_length=max_length,    # BGE-M3 上限 8192，这里先用 512 做快速测试
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        # outputs.last_hidden_state: [batch, seq_len, hidden_size]
        # 取 CLS 位置（第 0 个 token）
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # L2 归一化（常见做法，方便后面用内积当余弦相似度）
        cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)

        all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)  # [N, hidden_size]

def main():
    device = get_device()
    print("Using device:", device)

    tokenizer, model = load_bge_m3(device)

    # 随便准备几句多语言文本测试一下
    texts = [
        "这是一个中文句子，用来测试 BGE-M3 的中文嵌入效果。",
        "This is an English sentence to test BGE-M3 embeddings.",
        "Это русское предложение для проверки встраиваний BGE-M3.",
    ]

    embs = encode_texts(tokenizer, model, texts, device)
    print("Embeddings shape:", embs.shape)  # 预期: [3, 1024]
    print("First vector (first 8 dims):", embs[0][:8])
    print("Pairwise cosine (via dot product):")
    sims = embs @ embs.T
    print(sims)

if __name__ == "__main__":
    main()
