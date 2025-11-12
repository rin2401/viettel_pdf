# viettel_pdf

## Flow

- Extract: Deepseek-OCR -> Norm
- Search: BGE-m3 search on chunk Header 1 -> Get top docs
- QA: Qwen3-4B-Instruct-2507 get answer from each top docs

## Run

```bash
# Run install uv and venv
bash install_env.sh

# Run Deepseek-OCR (not optimizer ~ 7h)
bash run_extract.sh

# Run search and qa
bash run_qa.sh
```

## Env
- Docker Image: nvcr.io/nvidia/pytorch:23.08-py3 
- 1 GPU 24GB


## Score private
- Extract: 68
- QA: 78
- Overall: 73