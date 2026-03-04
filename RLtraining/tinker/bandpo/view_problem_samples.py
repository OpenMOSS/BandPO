import json
import pandas as pd

df = pd.read_parquet("/remote-home1/yli/Workspace/BandPO/data/dataset/math-500/train/train_L3-5.parquet")
n = 848 # 6017
row = df.iloc[n - 1].to_dict()

print(json.dumps(row, ensure_ascii=False, indent=2, default=str))

# {
#   "solution": "",
#   "subject": "Number Theory",
#   "level": 5,
#   "unique_id": "train/number_theory/7115.json",
#   "data_source": "math_dapo",
#   "prompt": "[{'content': 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\\n\\n For any integer $n>1$, the number of prime numbers greater than $n!+1$ and less than $n!+n$ is:\\n$\\\\text{(A) } 0\\\\quad\\\\qquad \\\\text{(B) } 1\\\\quad\\\\\\\\ \\\\text{(C) } \\\\frac{n}{2} \\\\text{ for n even, } \\\\frac{n+1}{2} \\\\text{ for n odd}\\\\quad\\\\\\\\ \\\\text{(D) } n-1\\\\quad \\\\text{(E) } n$ \\n\\nRemember to put your answer on its own line after \"Answer:\".', 'role': 'user'}]",
#   "ability": "MATH",
#   "reward_model": {
#     "ground_truth": "",
#     "style": "rule-lighteval\\/MATH_v2"
#   },
#   "extra_info": {
#     "answer": "",
#     "data_source": "nlile/hendrycks-MATH-benchmark",
#     "question_raw": "For any integer $n>1$, the number of prime numbers greater than $n!+1$ and less than $n!+n$ is:\n$\\text{(A) } 0\\quad\\qquad \\text{(B) } 1\\quad\\\\ \\text{(C) } \\frac{n}{2} \\text{ for n even, } \\frac{n+1}{2} \\text{ for n odd}\\quad\\\\ \\text{(D) } n-1\\quad \\text{(E) } n$",
#     "split": "train"
#   }
# }

# {
#   "solution": "",
#   "subject": "Number Theory",
#   "level": 5,
#   "unique_id": "train/number_theory/7117.json",
#   "data_source": "math_dapo",
#   "prompt": "[{'content': 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\\n\\n You are given a sequence of $58$ terms; each term has the form $P+n$ where $P$ stands for the product $2 \\\\times 3 \\\\times 5 \\\\times\\\\ldots \\\\times 61$ of all prime numbers less than or equal to $61$, and $n$ takes, successively, the values $2, 3, 4,\\\\ldots, 59$. Let $N$ be the number of primes appearing in this sequence. Then $N$ is:\\n$\\\\textbf{(A)}\\\\ 0\\\\qquad \\\\textbf{(B)}\\\\ 16\\\\qquad \\\\textbf{(C)}\\\\ 17\\\\qquad \\\\textbf{(D)}\\\\ 57\\\\qquad \\\\textbf{(E)}\\\\ 58$ \\n\\nRemember to put your answer on its own line after \"Answer:\".', 'role': 'user'}]",
#   "ability": "MATH",
#   "reward_model": {
#     "ground_truth": "",
#     "style": "rule-lighteval\\/MATH_v2"
#   },
#   "extra_info": {
#     "answer": "",
#     "data_source": "nlile/hendrycks-MATH-benchmark",
#     "question_raw": "You are given a sequence of $58$ terms; each term has the form $P+n$ where $P$ stands for the product $2 \\times 3 \\times 5 \\times\\ldots \\times 61$ of all prime numbers less than or equal to $61$, and $n$ takes, successively, the values $2, 3, 4,\\ldots, 59$. Let $N$ be the number of primes appearing in this sequence. Then $N$ is:\n$\\textbf{(A)}\\ 0\\qquad \\textbf{(B)}\\ 16\\qquad \\textbf{(C)}\\ 17\\qquad \\textbf{(D)}\\ 57\\qquad \\textbf{(E)}\\ 58$",
#     "split": "train"
#   }
# }