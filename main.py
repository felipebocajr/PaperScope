import arxiv
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter
import time
import logging

import boto3
import re

# Load environment
load_dotenv()

# logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Use the on-demand inference profile ID for Anthropic Claude Haiku 4.5
# (list inference profiles with `aws bedrock list-inference-profiles`)
DEFAULT_INFERENCE_PROFILE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ============================================================================
# ARXIV PAPER FETCHING
# ============================================================================

def arxiv_result_to_dict(r: arxiv.Result) -> dict:
    print(f"Debug - entry_id: {r.entry_id}")
    print(f"Debug - get_short_id: {r.get_short_id()}")
    print(f"Debug - pdf_url: {r.pdf_url}")
    
    return {
        "ID": r.get_short_id,
        "title": r.title,
        "summary": r.summary,
        "pdf_url": r.pdf_url
}


def fetch_arxiv_papers(query="cat:cs.AI", max_results=10, days_back=7):
    """Fetch papers from arXiv."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    client = arxiv.Client()
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    results = []
    for r in client.results(search):
        if r.published and r.published >= cutoff:
            results.append(arxiv_result_to_dict(r))
        else:
            break
    
    return results

# ============================================================================
# LLM EXTRACTION (CLAUDE VIA AWS BEDROCK)
# ============================================================================

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def build_extraction_prompt(paper: dict) -> str:
    title = paper.get("title", "")
    abstract = paper.get("summary", "")
    primary_category = paper.get("primary_category", "")

    # Stronger instruction + short example to encourage valid JSON-only responses
    return f"""Extract structured signals from this arXiv paper abstract.

Return ONLY valid JSON with these exact keys (no surrounding text):
- main_topic: string (2-6 words, broad area)
- methods: array of strings (0-6 short method names)
- keywords: array of strings (3-10 short technical terms)

Rules:
- Base extraction only on information explicitly in the abstract
- Preserve multi-word phrases (e.g., "reinforcement learning" stays together)
- Return empty arrays if uncertain
- Do not include any markdown or explanation, only raw JSON

Example JSON:
{{"main_topic": "graph neural networks", "methods": ["message passing"], "keywords": ["gnn", "graph representation", "node classification"]}}

Paper:
Category: {primary_category}
Title: {title}

Abstract:
{abstract}

JSON:""".strip()


def parse_json_from_model_text(text: str) -> dict:
    """
    claude usually returns clean JSON if instructed, but this makes it robust:
    it extracts the first {...} block if extra text slips in.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def extract_signals_with_bedrock(
    papers: list[dict],
    model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0",
    region: str = "us-east-1"
) -> list[dict]:
    """Extract signals using AWS Bedrock.

    This implementation decodes the response body, retries on transient errors,
    parses JSON returned by the model, normalizes values and logs progress.
    """

    bedrock = boto3.client('bedrock-runtime', region_name=region)
    signals: list[dict] = []

    for i, paper in enumerate(papers, start=1):
        prompt = build_extraction_prompt(paper)

        # simple retry loop (rate limits / transient errors)
        last_err = None
        for attempt in range(1, 6):
            try:
                response = bedrock.invoke_model(
                    modelId=model_id,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1000,
                        "messages": [{
                            "role": "user",
                            "content": prompt
                        }]
                    })
                )

                # response['body'] is a StreamingBody; read and decode
                raw = response.get('body')
                if raw is None:
                    raise RuntimeError('empty response body from Bedrock')

                try:
                    text = raw.read()
                    if isinstance(text, bytes):
                        text = text.decode('utf-8', errors='ignore')
                except Exception:
                    # fallback: attempt to convert directly
                    text = str(raw)

                # Parse Bedrock's nested response: extract text from content[0]["text"]
                try:
                    response_obj = json.loads(text)
                    if "content" in response_obj and len(response_obj["content"]) > 0:
                        actual_text = response_obj["content"][0].get("text", "")
                    else:
                        actual_text = text

                    # Remove markdown code blocks if present
                    actual_text = actual_text.strip()
                    if actual_text.startswith("```"):
                        # Remove ```json or ``` opening and closing ```
                        lines = actual_text.split("\n")
                        actual_text = "\n".join(lines[1:-1]) if len(lines) > 2 else actual_text

                    data = parse_json_from_model_text(actual_text)
                except Exception as e:
                    raise RuntimeError(f"failed to parse model JSON: {e}; raw={text[:500]}")

                # Build normalized signal
                def norm_str(v):
                    return normalize_text(str(v)) if v is not None else ""

                main_topic = norm_str(data.get('main_topic', ''))
                methods = [normalize_text(x) for x in (data.get('methods') or []) if str(x).strip()][:6]
                keywords = [normalize_text(x) for x in (data.get('keywords') or []) if str(x).strip()][:10]

                signal = {
                    "main_topic": main_topic,
                    "methods": methods,
                    "keywords": keywords,
                    "_title": paper.get('title'),
                    "_published": paper.get('published'),
                    "_primary_category": paper.get('primary_category'),
                }

                signals.append(signal)
                break

            except Exception as e:
                last_err = e
                logger.warning("Attempt %d failed for paper %d: %s", attempt, i, str(e))
                time.sleep(min(2 ** attempt, 10))
        else:
            # after retries, record a placeholder and continue
            logger.error("Failed to extract for paper %d after retries: %s", i, str(last_err))
            signals.append({
                "main_topic": "",
                "methods": [],
                "keywords": [],
                "_title": paper.get('title'),
                "_error": str(last_err),
            })

        # gentle pacing to reduce rate-limit risk
        time.sleep(0.15)

        if i % 10 == 0:
            logger.info("Extracted %d/%d papers", i, len(papers))

    logger.info("Extraction complete: %d signals", len(signals))
    return signals

# ============================================================================
# SIGNAL AGGREGATION
# ============================================================================

def normalize_text(s: str) -> str:
    """Normalize text: lowercase and strip whitespace."""
    return " ".join(s.lower().strip().split())


def top_items(counter: Counter, n: int = 10):
    """Get top n items from counter as list of dicts."""
    return [{"name": k, "count": v} for k, v in counter.most_common(n)]


def aggregate_signals(signals: list[dict], top_n: int = 15) -> dict:
    """Aggregate signals to find trends in papers."""
    topic_counter = Counter()
    method_counter = Counter()
    keyword_counter = Counter()

    for item in signals:
        topic = item.get("main_topic")
        if topic:
            topic_counter[normalize_text(topic)] += 1

        for m in item.get("methods", []) or []:
            method_counter[normalize_text(m)] += 1

        for kw in item.get("keywords", []) or []:
            keyword_counter[normalize_text(kw)] += 1

    # Build themes: union of top topics + top methods
    themes = []
    for k, _ in topic_counter.most_common(top_n):
        themes.append(k)
    for k, _ in method_counter.most_common(top_n):
        if k not in themes:
            themes.append(k)

    return {
        "topic_counts": top_items(topic_counter, top_n),
        "method_counts": top_items(method_counter, top_n),
        "keyword_counts": top_items(keyword_counter, top_n),
        "top_themes": themes[:top_n],
        "num_papers": len(signals),
    }


# ============================================================================
# FILE I/O
# ============================================================================

def save_json(data, filepath):
    """Save data to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return None
        return json.loads(content)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Fetch papers from arXiv
    logger.info("Fetching papers from arXiv...")
    start = time.time()
    papers = fetch_arxiv_papers()
    end = time.time()
    logger.info("✓ Fetched %d papers in %.2f seconds", len(papers), (end - start))
    
    # Save papers
    save_json(papers, "data/papers.json")
    logger.info("✓ Saved to data/papers.json")
    
    # Aggregate signals if available
    signals_path = Path("data/signals.json")
    if not signals_path.exists():
        logger.info("\nExtracting signals with Bedrock inference profile %s...", DEFAULT_INFERENCE_PROFILE)
        papers = load_json("data/papers.json")  # or reuse the `papers` variable already in memory
        signals = extract_signals_with_bedrock(papers, model_id=DEFAULT_INFERENCE_PROFILE)
        save_json(signals, "data/signals.json")
        logger.info("✓ Saved signals to data/signals.json")

