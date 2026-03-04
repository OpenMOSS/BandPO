#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from openai import OpenAI
import numpy as np

# 和你之前调用 DeepSeek 的方式一致
client = OpenAI(
    api_key="EMPTY",                           # 随便填
    base_url="http://10.176.58.103:8888/v1",   # 换成你 vLLM 服务的地址
)

def main():
    texts = [
        # "这是一个中文句子，用来测试 BGE-M3 在 vLLM 上的 embedding。",
        # "This is an English sentence to test BGE-M3 embeddings via vLLM.",
        # "Das ist ein deutscher Testsatz für BGE-M3.",
        # "Das ist ein deutscher Satz, um BGE-M3-Embeddings über vLLM zu testen.",
        "等一下，这里我好像有点算错了。我们在求解这个二次方程的时候，系数是 a=1, b=-5, c=6，对吧？如果是这样的话，判别式应该是 Δ = b^2 - 4ac = (-5)^2 - 4*1*6。先别着急往下算，我再检查一遍条件：题目说的是实数解，所以我们确实需要确保判别式是非负的。好，再来一次，Δ = 25 - 24 = 1，其实是大于 0 的。这就说明方程应该有两个不相等的实根。我刚刚担心的是会不会出现 Δ<0 的情况，就会导致没有实数解。但现在看来只是我自己多想了，原来的思路其实是对的，只是我在脑子里把 24 看成了 26，导致一瞬间以为判别式是负的。所以总结一下，等一下的这段检查其实只是确认：判别式是正的，确实存在两个实根，我们可以放心继续用求根公式往下算。",
        "Wait, I think I might have messed something up here. For this quadratic equation we have a = 1, b = -5, c = 6, right? If that’s correct, then the discriminant should be Δ = b^2 - 4ac = (-5)^2 - 4*1*6. Before I go further, let me double-check the requirements: the problem is asking for real solutions, so we really need Δ to be non-negative. Okay, let’s recompute carefully: Δ = 25 - 24 = 1, which is actually greater than 0. That means the equation has two distinct real roots. What I was worried about a moment ago was the case Δ < 0, which would imply no real solutions, but that was just me misreading the numbers. I briefly treated 24 as if it were 26 and got confused, so this whole “wait” section is basically me verifying that the discriminant is positive and that using the quadratic formula is still perfectly fine here.",
        "Moment mal, ich glaube, ich habe mich hier irgendwo vertan. Für diese quadratische Gleichung gelten doch a = 1, b = -5, c = 6, oder? Wenn das stimmt, dann ist die Diskriminante Δ = b² - 4ac = (-5)² - 4*1*6. Bevor ich weiterrechne, möchte ich kurz die Bedingung überprüfen: Die Aufgabe verlangt reelle Lösungen, also muss Δ größer oder gleich null sein. Rechnen wir das noch einmal ganz sauber: Δ = 25 - 24 = 1, das ist tatsächlich positiv. Das bedeutet, die Gleichung besitzt zwei verschiedene reelle Nullstellen. Vorhin hatte ich kurz Sorge, dass Δ < 0 sein könnte und es gar keine reellen Lösungen gibt, aber das lag nur daran, dass ich im Kopf 24 mit 26 verwechselt habe. Diese ganze „Moment mal“-Passage dient also nur dazu, zu bestätigen, dass die Diskriminante positiv ist und wir ganz normal mit der Mitternachtsformel weitermachen können.",
        "Hold on, I should pause and verify my calculation. For this quadratic we’re working with coefficients a = 1, b = -5, and c = 6. Under those values the discriminant is supposed to be Δ = b^2 - 4ac = (-5)^2 - 4*1*6. Since the question is explicitly asking for real roots, everything depends on whether this discriminant is non-negative. Let me go through it again carefully: Δ = 25 - 24, which comes out to 1, so it’s clearly positive. That immediately tells us the equation has two distinct real solutions. The confusion I had a moment ago came from a mental slip where I treated 24 as if it were 26 and briefly concluded that Δ would be negative. In other words, this whole “hold on” detour is just me rechecking the numbers and confirming that the discriminant is actually positive and that the original approach with the quadratic formula is completely valid.",
    ]

    # 注意：这里的 model 名必须和 vLLM 启动时的 --served-model-name 对齐
    resp = client.embeddings.create(
        model="BAAI/bge-m3",   # 如果你 serve 时写的是 --served-model-name bge-m3，就改成 "bge-m3"
        input=texts,
    )

    embeddings = [item.embedding for item in resp.data]
    embeddings = np.array(embeddings, dtype=np.float32)  # [N, D]

    print("num embeddings:", embeddings.shape[0])
    print("embedding dim:", embeddings.shape[1])

    for i, text in enumerate(texts):
        emb = embeddings[i]
        print(f"\nText {i}: {text}")
        print("  first 8 dims:", emb[:8])

    # ---- 计算余弦相似度矩阵 ----
    # 先做 L2 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings_norm = embeddings / norms

    # 余弦相似度 = 归一化向量的内积
    sim_matrix = embeddings_norm @ embeddings_norm.T   # [N, N]

    print("\nCosine similarity matrix:")
    # 打印成 3x3 的小矩阵
    with np.printoptions(precision=4, suppress=True):
        print(sim_matrix)

    # 单独再把两两之间的相似度打印一下
    print("\nPairwise cosine similarities:")
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            print(f"sim(text{i}, text{j}) = {sim_matrix[i, j]:.4f}")

if __name__ == "__main__":
    main()
