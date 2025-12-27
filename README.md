# Corrective RAG (OpenAI + PDF)

Cost-optimized Corrective RAG using a persisted FAISS index and OpenAI models.

## Overview

- Corrective RAG pipeline that evaluates retrieved context, refines queries when needed, and produces answers.
- Uses a persisted FAISS index under `FAISS_store/` to avoid re-embedding.

## Quickstart

1. Create a Python 3.10+ virtual environment and activate it:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

4. Place PDF sources in `sources/` (example: `sources/Third-Avartana-opens-at-ITC-M.pdf`).

5. Run the main script:

```bash
python index.py
```

## Files of interest

- `index.py`: main entrypoint — exposes `corrective_rag()` pipeline and creates/loads FAISS index.
- `requirements.txt`: project dependencies.
- `FAISS_store/`: persisted FAISS indices (should be ignored by git).

## Notes

- Adjust `PDF_PATH`, chunking parameters, and `SIMILARITY_SKIP_THRESHOLD` inside `index.py` as needed.
- The script prints a warning if `OPENAI_API_KEY` is not set; ensure you create `.env` or set the env var.

## License

- Add a license file if you plan to make this public.
