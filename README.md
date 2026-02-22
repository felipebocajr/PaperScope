# PaperScope

A simple Python application for collecting research papers from arXiv and viewing them in a web UI.

## Project Structure

```
paperscore/
├── main.py               # Core logic: fetch papers, aggregate signals
├── streamlit_app.py      # Web UI for viewing papers
├── api_test.py           # Test Gemini API integration
├── requirements.txt      # Dependencies
├── .env                  # Environment variables (API keys)
├── data/
│   └── papers.json       # Collected papers
└── README.md             # This file
```

## Features

- **Fetch Papers**: Collects recent research papers from arXiv via API
- **Organize**: Structures papers with metadata (title, authors, categories, abstract)
- **View**: Web interface to browse and filter papers by category
- **Analyze**: Optional signal aggregation to identify research trends
- **API Integration**: Test Gemini AI API for potential features

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API key (optional, only for `api_test.py`):
```bash
export GEMINI_API_KEY='your-api-key-here'
```

## Usage

### Collect papers from arXiv
```bash
python main.py
```

This fetches papers from the last 7 days in the AI category and saves them to `data/papers.json`.

### View papers in web UI
```bash
streamlit run streamlit_app.py
```

Browse, filter, and view papers in an interactive dashboard.

### Test API
```bash
python api_test.py
```

Tests Gemini API connection (requires `GEMINI_API_KEY` environment variable).

## Configuration

Edit `main.py` to customize:
- `fetch_arxiv_papers()` - Query, max results, days back
- `aggregate_signals()` - Top N items to track
- File paths for saving data

### S3 Upload (optional)
The pipeline can automatically upload the generated weekly digest JSON to an S3 bucket. Set the following environment variables in your `.env` or shell:

```bash
# Leave empty to disable S3 upload
OUTPUT_BUCKET=ai-research-digest

# Optional prefix/folder inside the bucket (e.g. "weeks")
OUTPUT_PREFIX=weeks
```

If `OUTPUT_BUCKET` is not set or empty, the upload step is skipped. The object key will be `<prefix>/<filename>` if a prefix is provided, otherwise just the filename.

## Code Organization

The code is organized by function groups:

**main.py:**
- `ARXIV PAPER FETCHING` - Download papers from arXiv
- `SIGNAL AGGREGATION` - Analyze trends in papers
- `FILE I/O` - Save/load JSON data
- `MAIN` - Orchestrate the workflow

**streamlit_app.py:**
- Simple app that loads and displays papers

**api_test.py:**
- Tests Gemini API with safe error handling

## Example Customization

Fetch different category and time range:
```python
papers = fetch_arxiv_papers(
    query="cat:cs.LG",  # Machine Learning
    max_results=500,
    days_back=14
)
```

Filter papers programmatically:
```python
cs_ai_papers = [p for p in papers if "cs.AI" in p["categories"]]
```

## Dependencies

- **arxiv** - arXiv API client
- **streamlit** - Web UI framework  
- **google-genai** - Gemini API client
- **requests** - HTTP library

See `requirements.txt` for exact versions.

