from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://10.176.58.103:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
instruction="Find the matrix $\\mathbf{M}$ such that\n\\[\\mathbf{M} \\begin{pmatrix} 1 & -2 \\\\ 1 & 4 \\end{pmatrix} = \\begin{pmatrix} 6 & 0 \\\\ 0 & 6 \\end{pmatrix}.\\]"
prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
resp = client.chat.completions.create(model="/remote-home1/yli/Workspace/DiagVerse/data/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",     
    messages=[
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": instruction},
    ],
    temperature=0.2,
    )
print("Chat result:", resp.choices[0].message.content)
import pdb
pdb.set_trace()

