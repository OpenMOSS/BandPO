import pandas as pd

IN_PATH = "/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/train/train_L3-5.parquet"
OUT_PATH = IN_PATH.replace(".parquet", "_fixed.parquet")

# 我们要修的两道题（用 unique_id + 原题面做确认）
targets = {
    "train/number_theory/7115.json": {
        "question_raw": r"For any integer $n>1$, the number of prime numbers greater than $n!+1$ and less than $n!+n$ is:"
                        "\n"
                        r"$\text{(A) } 0\quad\qquad \text{(B) } 1\quad\\ \text{(C) } \frac{n}{2} \text{ for n even, } \frac{n+1}{2} \text{ for n odd}\quad\\ \text{(D) } n-1\quad \text{(E) } n$",
        "answer": "0",
        "solution": "For k=2,3,...,n-1 we have k|n!, so n!+k is divisible by k and is >k, hence composite. Therefore there are no primes in (n!+1, n!+n).\nAnswer: 0",
    },
    "train/number_theory/7117.json": {
        "question_raw": r"You are given a sequence of $58$ terms; each term has the form $P+n$ where $P$ stands for the product $2 \times 3 \times 5 \times\ldots \times 61$ of all prime numbers less than or equal to $61$, and $n$ takes, successively, the values $2, 3, 4,\ldots, 59$. Let $N$ be the number of primes appearing in this sequence. Then $N$ is:"
                        "\n"
                        r"$\textbf{(A)}\ 0\qquad \textbf{(B)}\ 16\qquad \textbf{(C)}\ 17\qquad \textbf{(D)}\ 57\qquad \textbf{(E)}\ 58$",
        "answer": "0",
        "solution": "Let p be any prime divisor of n (2<=n<=59). Then p<=59<=61 so p|P and p|n, hence p|(P+n). Since P+n>p, each term is composite. Thus N=0.\nAnswer: 0",
    },
}

df = pd.read_parquet(IN_PATH)

for uid, patch in targets.items():
    rows = df.index[df["unique_id"] == uid].tolist()
    if len(rows) != 1:
        raise RuntimeError(f"unique_id={uid} 找到 {len(rows)} 行（期望 1 行）: {rows}")

    i = rows[0]
    row = df.loc[i]

    # --- 1) 先确认：题面是我们要修的那道 ---
    q = row["extra_info"].get("question_raw", "")
    if (q or "").strip() != patch["question_raw"].strip():
        raise RuntimeError(f"unique_id={uid} 的 question_raw 不匹配，拒绝修改。\n"
                           f"实际: {q[:120]}...\n"
                           f"期望: {patch['question_raw'][:120]}...")

    # --- 2) 再确认：三个答案位置确实都为空（缺答案样本）---
    sol_empty = (row["solution"] == "")
    gt_empty = (row["reward_model"].get("ground_truth", "") == "")
    ans_empty = (row["extra_info"].get("answer", "") == "")

    if not (sol_empty and gt_empty and ans_empty):
        raise RuntimeError(
            f"unique_id={uid} 不是“答案全空”的样本，拒绝修改。\n"
            f"solution empty={sol_empty}, ground_truth empty={gt_empty}, extra_info.answer empty={ans_empty}"
        )

    # --- 3) 修改三个位置 ---
    df.at[i, "solution"] = patch["solution"]

    rm = dict(row["reward_model"])
    rm["ground_truth"] = patch["answer"]
    df.at[i, "reward_model"] = rm

    ex = dict(row["extra_info"])
    ex["answer"] = patch["answer"]
    df.at[i, "extra_info"] = ex

    print(f"Patched {uid} at row index {i}: answer={patch['answer']}")

df.to_parquet(OUT_PATH, index=False)
print(f"Saved to: {OUT_PATH}")
