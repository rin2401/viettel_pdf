# pip install transformers==4.47.1

import os

from transformers import AutoModel, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)


import fitz
import os
from tqdm.auto import tqdm
from PIL import Image
from io import BytesIO
import re
import os
import glob


def norm_data(data):
    # data = re.sub(r"<table>.*?</table>", "", data)

    # data = data.replace("<table>", "<table><thead></thead><tbody>").replace("</table>", "</tbody></table>")

    data = re.sub(
        r"<table>.*?(Lần ban hành|VIETTEL AI RACE).*?</table>",
        "",
        data,
        count=0,
        flags=0,
    )
    data = re.sub(r"VIETTEL AI RACE.*?Lần ban hành: 1", "", data, flags=re.DOTALL)

    imgs = re.findall(r"!\[\]\(images/.*?.jpg\)", data)

    i = 0
    for m in imgs:
        idx = data.find(m)
        # print(data[idx: idx+50])

        if "[Hình" in data[idx : idx + 50]:
            data = data.replace(m, f"|<image_{i+1}>|", 1)
            i += 1
        else:
            data = data.replace(m, "", 1)

    data = re.sub(r"</table>[ \n]*<table>", "", data, flags=re.DOTALL)

    data = re.sub(r"\n#+?(\d+)\. ", r"\n#\1\. ", data, flags=re.DOTALL)

    data = re.sub(r"\n#+ ([^\d])", r"\n\1", data, flags=re.DOTALL)

    data = re.sub(r"\n#* ?(\d+\.) ", r"\n# ", data, flags=re.DOTALL)
    data = re.sub(r"\n#* ?(\d+\.\d+) ", r"\n## ", data, flags=re.DOTALL)
    data = re.sub(r"\n#* ?(\d+\.\d+\.\d+) ", r"\n### ", data, flags=re.DOTALL)
    # data = re.sub(r"\n#* ?Hình (\d+)", r"\nHình \1", data, flags=re.DOTALL)

    # v18
    # data = re.sub(r"\n#* ?(\d+\.\d+\.\d+\.\d+\.?) ", r"\n#### ", data, flags=re.DOTALL)
    # tables = re.findall(r"<table>.*?</table>", data, flags=re.DOTALL)
    # for table in tables:
    #     # table2 = table.replace("<td>", "<td><blockquote><p>").replace("</td>", "</p></blockquote></td>")

    #     rows = re.findall(r"<tr>.*?</tr>", table, flags=re.DOTALL)
    #     head = rows[0].replace("<td>", "<th>").replace("</td>", "</th>")
    #     body = "".join(rows[1:])
    #     new_table =  f"<table><thead>{head}</thead><tbody>{body}</tbody></table>".replace("<", "\n<")
    #     data = data.replace(table, new_table, 1)

    return data


def parse_pdf(path):
    doc = fitz.open(path)

    md_pages = []
    for i in tqdm(range(len(doc))):
        page = doc.load_page(i)
        pix = page.get_pixmap()
        img = Image.open(BytesIO(pix.tobytes("png")))

        image_file = "./tmp/page.jpg"
        output_path = "./tmp"

        os.makedirs(output_path, exist_ok=True)

        img.save(image_file)

        # prompt = "<image>\nFree OCR. "
        prompt = "<image>\n<|grounding|>Convert the document to markdown. "
        # image_file = 'vng.jpg'
        # output_path = './vng'

        # infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

        # Tiny: base_size = 512, image_size = 512, crop_mode = False
        # Small: base_size = 640, image_size = 640, crop_mode = False
        # Base: base_size = 1024, image_size = 1024, crop_mode = False
        # Large: base_size = 1280, image_size = 1280, crop_mode = False

        # Gundam: base_size = 1024, image_size = 640, crop_mode = True

        res = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_file,
            output_path=output_path,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=True,
        )

        with open(output_path + "/result.mmd", encoding="utf-8") as f:
            md = f.read()

            md_pages.append(md)

    name = path.split("/")[-1].replace(".pdf", "")

    output = f"../dsocr_private/{name}/main.md"

    os.makedirs(output.rsplit("/", 1)[0], exist_ok=True)

    data = "\n\n".join(md_pages)

    name = path.split("/")[-1].split(".")[0]
    data = f"# {name}\n\n" + data

    with open(output, "w", encoding="utf-8") as f:
        f.write(data)


import glob

paths = glob.glob("../private_test_input/*.pdf")

for path in paths:
    parse_pdf(path)
