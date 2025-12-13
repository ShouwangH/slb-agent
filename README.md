# SLB Agent

Real Estate Funding Workflow Agent - Sale-Leaseback Template

## Quick Start

### Backend

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install ".[dev]"

# Run the API server
uvicorn app.api:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at http://localhost:5173 and proxies `/api` requests to the backend.

## Environment Variables

Copy `.env.example` to `.env`:

- `OPENAI_API_KEY`: If set, uses real OpenAI API. If empty, uses MockLLMClient for testing.

## Testing

```bash
pytest tests/ -v
```
