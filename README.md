# SOP Compliance Verification Engine

This tool compares a user-selected SOP against a set of Reference Documents to checks for compliance gaps.

## Configuration
1. Rename `.env.example` to `.env`.
2. Add your API key:
   ```bash
   GEMINI_API_KEY=AIzaSy...
   ```

## Prerequisites
- Python 3.8+
- **Google Gemini API Key** (Set in `.env`)
- `pip install -r requirements.txt`

## Architecture
- **Ingestion**: Parses Reference PDFs provided by user, normalizes them into atomic requirements using **Google Gemini (1.5 Flash)**, embeds them using SentenceTransformers (all-MiniLM-L6-v2), and stores them in S3/Local.
- **Verification**: Loads the Index, ingests the target SOP (ephemeral), and performs a "Reverse Check" using Semantic Search + **Google Gemini Validation**.

## Usage

### 1. Ingest Reference Documents (One-time)
Run this to build your "Truth" index in S3.
```bash
python -m src.main ingest --files "./refs/GCP_Guide.pdf" "./refs/ICH_E6.docx" --bucket "your-s3-bucket-name"
```

### 2. Verify an SOP
Run this to check an SOP against the indexed references.
```bash
python -m src.main verify --sop "./sops/My_SOP_v1.pdf" --bucket "your-s3-bucket-name" --output "report.json"
```

## Output
The tool will output a JSON report (`gap_report.json`) containing:
- Requirement ID
- Status (MISSING, PARTIAL, WEAK)
- Justification (from LLM)
- Source Reference Text
