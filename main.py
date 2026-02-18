import arxiv
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
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

DEFAULT_INFERENCE_PROFILE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ============================================================================
# ARXIV PAPER FETCHING
# ============================================================================

def arxiv_result_to_dict(r: arxiv.Result) -> dict:
    """Convert arxiv.Result to dictionary."""
    return {
        "id": r.get_short_id(),
        "title": r.title,
        "summary": r.summary,
        "published": r.published.strftime("%Y-%m-%d") if r.published else None,
        "pdf_url": r.pdf_url,
    }


def fetch_arxiv_papers(query="cat:cs.AI", max_results=20, days_back=7):
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
# PHASE 1: GENERIC SCORING (title + abstract)
# ============================================================================

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_scoring_prompt(paper: dict) -> str:
    """Build prompt for generic paper scoring based on title and abstract."""
    title = paper.get("title", "")
    abstract = paper.get("summary", "")

    return f"""You are a data science expert curating the most interesting AI/ML research papers for a LinkedIn audience of data scientists and engineers.

IMPORTANT: You are evaluating recently published research from {datetime.now().strftime('%B %Y')}. Your training data may be outdated. Do not penalize papers for mentioning tools, models, or techniques that are newer than your knowledge cutoff. Focus on the research contribution itself.

Score this paper with a single score from 0 to 10, considering:
- Innovation: Does it challenge current assumptions or introduce a genuinely new approach?
- Impact: Does it show significant performance improvements or strong real-world potential?
- Methodology: Is the method solid, reproducible and clearly explained?

Return ONLY valid JSON with exactly these keys:
{{"score": <number 0-10>, "reasoning": "<one sentence explaining the score>"}}

Paper:
Title: {title}

Abstract:
{abstract}

JSON:""".strip()


def parse_json_from_model_text(text: str) -> dict:
    """Extract the first JSON object from model text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def invoke_bedrock(bedrock_client, model_id: str, prompt: str) -> str:
    """Call Bedrock and return the text response."""
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    raw = response.get("body")
    if raw is None:
        raise RuntimeError("Empty response body from Bedrock")

    text = raw.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    response_obj = json.loads(text)
    actual_text = response_obj.get("content", [{}])[0].get("text", "")

    # Strip markdown code blocks if present
    actual_text = actual_text.strip()
    if actual_text.startswith("```"):
        lines = actual_text.split("\n")
        actual_text = "\n".join(lines[1:-1]) if len(lines) > 2 else actual_text

    return actual_text


def score_papers_with_bedrock(
    papers: list[dict],
    model_id: str = DEFAULT_INFERENCE_PROFILE,
    region: str = "us-east-1",
) -> list[dict]:
    """
    Phase 1: Score each paper (title + abstract) with a generic 0-10 score.
    Returns a list of dicts with id, score, and reasoning.
    """
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    scores: list[dict] = []

    for i, paper in enumerate(papers, start=1):
        prompt = build_scoring_prompt(paper)

        last_err = None
        for attempt in range(1, 6):
            try:
                actual_text = invoke_bedrock(bedrock, model_id, prompt)
                data = parse_json_from_model_text(actual_text)

                scores.append({
                    "id": paper.get("id"),
                    "score": float(data.get("score", 0)),
                    "reasoning": data.get("reasoning", ""),
                    "_title": paper.get("title"),
                })
                break

            except Exception as e:
                last_err = e
                logger.warning("Attempt %d failed for paper %d: %s", attempt, i, str(e))
                time.sleep(min(2 ** attempt, 10))
        else:
            logger.error("Failed to score paper %d after retries: %s", i, str(last_err))
            scores.append({
                "id": paper.get("id"),
                "score": 0.0,
                "reasoning": "",
                "_title": paper.get("title"),
                "_error": str(last_err),
            })

        time.sleep(0.15)

        if i % 10 == 0:
            logger.info("Scored %d/%d papers", i, len(papers))

    logger.info("Scoring complete: %d papers scored", len(scores))
    return scores


def get_top_papers(papers: list[dict], scores: list[dict], top_n: int = 10) -> list[dict]:
    """
    Pick the top N papers by score.
    Returns the full paper dicts (including pdf_url) for the top N.
    """
    # Build a lookup: id -> full paper dict
    papers_by_id = {p["id"]: p for p in papers}

    # Sort scores descending
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)

    top = []
    for s in sorted_scores[:top_n]:
        paper = papers_by_id.get(s["id"])
        if paper:
            # Attach score and reasoning to the paper dict
            paper["score"] = s["score"]
            paper["reasoning"] = s["reasoning"]
            top.append(paper)

    return top


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

    # -------------------------------------------------------------------------
    # Step 1: Fetch papers
    # -------------------------------------------------------------------------
    logger.info("Fetching papers from arXiv...")
    start = time.time()
    papers = fetch_arxiv_papers()
    logger.info("Fetched %d papers in %.2f seconds", len(papers), time.time() - start)
    save_json(papers, "data/papers.json")
    logger.info("Saved to data/papers.json")

    # -------------------------------------------------------------------------
    # Step 2: Score all papers (Phase 1 - generic score)
    # -------------------------------------------------------------------------
    scores_path = Path("data/scores.json")
    if not scores_path.exists():
        logger.info("Scoring papers with Bedrock...")
        scores = score_papers_with_bedrock(papers)
        save_json(scores, "data/scores.json")
        logger.info("Saved scores to data/scores.json")
    else:
        logger.info("Scores already exist, loading from data/scores.json")
        scores = load_json("data/scores.json")

    # -------------------------------------------------------------------------
    # Step 3: Pick top 10 papers
    # -------------------------------------------------------------------------
    top_papers = get_top_papers(papers, scores, top_n=10)
    save_json(top_papers, "data/top_papers.json")
    logger.info("Top 10 papers saved to data/top_papers.json")

    for i, p in enumerate(top_papers, 1):
        logger.info("%d. [%.1f] %s", i, p["score"], p["title"])