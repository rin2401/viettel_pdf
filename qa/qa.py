import os

os.environ["VLLM_USE_V1"] = "0"

import pandas as pd
from vllm import LLM, SamplingParams

model_name = "/data01/kilm/users/quocvh/HF/models/Qwen/Qwen3-4B-Instruct-2507"
# model_name = "Qwen/Qwen3-4B-Instruct-2507"

llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.85,
    max_model_len=31000,
    enforce_eager=True,
)
tokenizer = llm.get_tokenizer()


SET = "private"
df = pd.read_csv(f"../qa_{SET}_search.csv")

# df.head()

data = df.to_dict("records")


TEMPLATE = """Tài liệu: {context}

Hãy chọn các đáp án đúng cho câu hỏi trắc nghiệm sau dựa vào tài liệu
Ouput format: 
Please show your choice in the answer field with only the choice letters, e.g., "C" or "A,D"

Câu hỏi: {Question}
A. {A}
B. {B}
C. {C}
D. {D}

Đáp án đúng là:
"""

import glob

md_paths = sorted(glob.glob(f"../dsocr_{SET}/*/*.md"))

md_docs = {}
for i, path in enumerate(md_paths):
    with open(path) as f:
        text = f.read()

    # print(len(text.split()))

    idx = int(path.split("/")[-2].split("_")[-1])
    # idx2i[idx] = i
    md_docs[idx] = text


import json
import ast

for i in range(10):
    na_data = [x for x in data if not x.get("pred")]
    print("Top", i, len(na_data))

    MM = []
    for x in na_data:
        idxs = ast.literal_eval(x["search"])
        # print(idxs)
        if i > len(idxs) - 1:
            x["pred"] = "A"
            continue

        idx = int(idxs[i].split("_")[-1])
        # print(idx)
        context = md_docs[idx]
        x["context"] = context

        x["context"] = tokenizer.decode(tokenizer.encode(x["context"])[:30000])
        prompt = TEMPLATE.format(**x)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        # MM.append(messages)

        pp = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        # pp += "\nanswer:"
        MM.append(pp)

    sampling_params = SamplingParams(temperature=0.0, top_k=1, max_tokens=10)
    outputs = llm.generate(MM, sampling_params)
    # preds = []
    for x, d in zip(outputs, na_data):
        res = x.outputs[0].text
        res = res.replace("answer:", "").strip()
        answers = []
        for a in res.split(","):
            a = a.split(".")[0].strip()

            if a:
                a = a[0]
            else:
                continue

            if a in "ABCD":
                answers.append(a)

        print(res, answers)

        if not answers:
            # preds.append("A")
            d["pred"] = None
        else:
            # preds.append(",".join(answers))
            d["pred"] = ",".join(answers)
    print("=" * 20)


preds = []
for x in data:
    if x["pred"]:
        preds.append(x["pred"])
    else:
        preds.append("A")
len(preds)


df["num_correct"] = [len(x.split(",")) for x in preds]
df["answers"] = preds
df.to_csv(f"qa_{SET}.csv", encoding="utf-8-sig", index=False)

subdf = df[["num_correct", "answers"]]


import os
import glob

INPUT_DIR = "../private_test_input"
OUTPUT_DIR = "../dsocr_private"

answer_md = os.path.join(OUTPUT_DIR, "answer.md")

with open(answer_md, "w", encoding="utf-8") as f:
    f.write("### TASK EXTRACT\n")
    for pdf_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.pdf"))):
        pdf_name = pdf_path.split("/")[-1].replace(".pdf", "")
        with open(
            os.path.join(OUTPUT_DIR, pdf_name, "main.md"),
            "r",
            encoding="utf-8",
        ) as md:
            f.write(md.read())
            f.write("\n")

    f.write("\n\n")
    f.write("### TASK QA\n")
    f.write(subdf.to_csv(index=False))

import shutil

shutil.copy("qa.py", OUTPUT_DIR)
shutil.make_archive(OUTPUT_DIR, "zip", OUTPUT_DIR)
