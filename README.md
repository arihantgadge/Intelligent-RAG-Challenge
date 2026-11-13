# Zyro Dynamics HR Help Desk (RAG)

A chatbot that answers employee HR questions from Zyro Dynamics' 11 internal policy PDFs, cites the policy section behind each answer, and politely refuses anything the documents do not cover.

## How it works

1. **Load and clean** – `pypdf` reads each PDF; repeated page headers and cover blocks are dropped and extraction artefacts are fixed.
2. **Chunk** – text is split by policy section (the ALL-CAPS headings), then into 900-character chunks with 150 overlap. Each chunk is prefixed with its document and section name so flattened table rows stay meaningful.
3. **Embed and index** – `BAAI/bge-small-en-v1.5` embeddings (via `fastembed`, no PyTorch needed) stored in FAISS.
4. **Retrieve** – MMR search (`k=8`, `fetch_k=24`, `lambda=0.8`).
5. **Guardrail** – an LLM scope check refuses out-of-scope questions before retrieval; the answer prompt adds a second check that refuses when the retrieved excerpts do not contain the answer.
6. **Generate** – Groq `openai/gpt-oss-120b` at temperature 0 writes a short, grounded, plain-text answer.

The whole flow is one LCEL chain in `rag_pipeline.py`, traced to LangSmith project `zyro-rag-challenge` when `LANGCHAIN_API_KEY` is set.

## Files

| File | Purpose |
| --- | --- |
| `rag_pipeline.py` | The RAG pipeline (shared by the app and the notebook) |
| `app.py` | Streamlit chat app with source citations |
| `Starter Notebook.ipynb` | Kaggle notebook: pipeline, chunk inspection (Step 11), retrieval settings experiment (Step 12), and `submission.csv` |
| `zyro-dynamics-hr-corpus/` | The 11 HR policy PDFs |

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   # then add your keys
streamlit run app.py
```

Keys: `GROQ_API_KEY` (required, free at console.groq.com) and `LANGCHAIN_API_KEY` (optional, free at smith.langchain.com). The app also accepts a Groq key typed into the sidebar.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (`.streamlit/secrets.toml` is git-ignored; never commit keys).
2. At share.streamlit.io choose **New app**, pick the repository and set the main file to `app.py`.
3. Under **Advanced settings > Secrets**, paste the contents of `.streamlit/secrets.toml.example` with your real keys.

## Generate the submission

Upload `Starter Notebook.ipynb` to Kaggle, attach the competition dataset, turn Internet on, add `GROQ_API_KEY` and `LANGCHAIN_API_KEY` under **Add-ons > Secrets**, and run all cells. The last cell writes `submission.csv`.
