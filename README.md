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
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key  # optional
   PINECONE_ENVIRONMENT=your_pinecone_env  # optional
   ```

### Run the audit
```bash
python red_team_audit.py --provider openai --vector-db faiss --model gpt-3.5-turbo --embed-model text-embedding-3-small --output trust_report.json
```

### Custom prompt list
```bash
python red_team_audit.py --prompt-file custom_prompts.txt --provider openai --vector-db faiss
```

### Provider options
- `--provider openai` (default)
- `--provider anthropic` (requires ANTHROPIC_API_KEY)
- `--provider azure` (requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)

### Vector DB options
- `--vector-db faiss` (local, default)
- `--vector-db milvus` (placeholder, requires pymilvus; fallback to Noop in this version)
- `--vector-db pinecone` (placeholder, requires pinecone-client; fallback to Noop in this version)

### Output
- `trust_score_report.json` includes per-attack outcomes and an aggregated trust score (0-100).

