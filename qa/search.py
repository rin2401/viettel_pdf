import re
import glob
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownHeaderTextSplitter
from sklearn.metrics.pairwise import cosine_similarity


model_name = "BAAI/bge-m3"

model = SentenceTransformer(model_name)


def split_md(doc):
    headers_to_split_on = [
        ("#", "Header 1"),
        # ("##", "Header 2"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    res = header_splitter.split_text(doc)
    chunks = ["# " + x.metadata["Header 1"] + "\n\n" + x.page_content for x in res]
    return chunks


SET = "private"
# SET = "public"

md_paths = sorted(glob.glob(f"../dsocr_{SET}/*/*.md"))
md_docs = {}
for i, path in enumerate(md_paths):
    with open(path) as f:
        text = f.read()

    idx = int(path.split("/")[-2].split("_")[-1])
    md_docs[idx] = text


raw_paths = sorted(glob.glob(f"../dsocr_{SET}/*/*.md"))

paths = []
docs = []
raw_docs = {}

name2i = {}

for i, path in enumerate(raw_paths):
    with open(path) as f:
        text = f.read()
    idx = int(path.split("/")[-2].split("_")[-1])
    raw_docs[idx] = text

    chunks = split_md(text)
    for c in chunks:
        paths.append(path)
        docs.append(c)

docs_embedding = model.encode(docs, batch_size=16)


df = pd.read_csv(f"../{SET}_test_input/question.csv")

contexts = []
search_results = []
c = 0
for x in df.to_dict("records"):
    m = re.findall(r"(?:TD|Public) ?_?(\d+)", x["Question"])
    print(m)
    if m and int(m[0]) in raw_docs:
        idx = int(m[0])

        print("Query has:", idx, x["Question"])
        contexts.append(raw_docs[idx])

        search_results.append([f"Public_{m[0]}"])
        c += 1
        continue

    query = """{Question}
{A}
{B}
{C}
{D}""".format(
        **x
    )

    queries = [query]
    # queries = [x["Question"]]

    # queries = [x["Question"], x["Question"] + " " + str(x["A"]), x["Question"]+ " " + str(x["B"]),  x["Question"]+ " " + str(x["C"]), x["Question"]+ " " + str(x["D"])]

    query_embedding = model.encode(queries)
    similarities = cosine_similarity(query_embedding, docs_embedding)

    # similarities = np.max(similarities, axis=0)
    similarities = np.mean(similarities, axis=0)
    # print(similarities.shape)

    idxs = np.argsort(similarities)[::-1]
    # print(idxs)
    names = []
    for idx in idxs:
        name = paths[idx].split("/")[-2]
        if name not in names:
            names.append(name)

    names = names[:10]
    # names = [paths[idx].split("/")[-2] for idx in idxs]
    # print(paths[idx], similarities[0][idx])
    search_results.append(names)

    idx = idxs[0]
    doc_idx = int(paths[idx].split("/")[-2].split("_")[-1])
    print(idx, doc_idx)

    contexts.append(md_docs[doc_idx])


df["search"] = search_results
df["context"] = contexts

df.to_csv(f"../qa_{SET}_search.csv", encoding="utf-8-sig", index=False)
