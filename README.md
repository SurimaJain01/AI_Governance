# AI_Governance

## LLM Red-Teaming Financial Auditor

This repository contains a prototype tool that tests a financial LLM for adversarial abuse and safety bypass attempts.

### Features
- Attack prompt library simulating prohibited financial advice and prompt injection
- Response safety detectors (prohibited keywords/patterns)
- Trust score report generation
- Optional vector store persistence (FAISS local by default; Milvus/Pinecone optional)

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with:
   ```bash
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_api_key  # required for embeddings
   ANTHROPIC_API_KEY=your_anthropic_api_key  # optional
   PINECONE_API_KEY=your_pinecone_api_key  # optional
   PINECONE_ENVIRONMENT=your_pinecone_env  # optional
   ```

### Run the audit
```bash
python red_team_audit.py --provider groq --model llama3-70b-8192 --embed-model text-embedding-3-small --output trust_report.json
```

### Custom prompt list
```bash
python red_team_audit.py --prompt-file custom_prompts.txt --provider groq --vector-db faiss
```

### Provider options
- `--provider groq` (default, requires GROQ_API_KEY)
- `--provider openai` (requires OPENAI_API_KEY)
- `--provider anthropic` (requires ANTHROPIC_API_KEY)
- `--provider azure` (requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)

### Vector DB options
- `--vector-db faiss` (local, default)
- `--vector-db milvus` (placeholder, requires pymilvus; fallback to Noop in this version)
- `--vector-db pinecone` (placeholder, requires pinecone-client; fallback to Noop in this version)

### Output
- `trust_score_report.json` includes per-attack outcomes and an aggregated trust score (0-100).

