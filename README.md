# Mental Health Support LLM (research prototype)

Streamlit demo comparing Qwen2.5-1.5B-Instruct: Base vs QLoRA SFT vs SFT -> ORPO.
Not a diagnostic or clinical tool; not clinically validated. Evaluation is preliminary
(only 5 Base responses fully rated) - do not claim SFT/ORPO outperforms Base.

## Layout
- `app.py`, `requirements.txt`, `render.yaml` - the deployed app (must stay at repo root)
- `models/` - adapter folders (see `models/README.md`)
- `notebooks/`, `evaluation/`, `docs/` - training workflow, saved comparisons, overview

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Render
1. Push this repo to GitHub (keep it private).
2. Render -> New -> Blueprint -> pick the repo -> Apply.
3. Use plan **Pro Plus (8 GB)**. Render web services have no GPU; the model needs ~3-4 GB RAM on CPU.
4. Open the URL; the first request downloads the base model (~3 GB) and is slow.
