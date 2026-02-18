import arxiv
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time
import logging
from io import BytesIO

import boto3
import requests
import re

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

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


def fetch_arxiv_papers(query="cat:cs.AI", max_results=25, days_back=7):
    """Fetch papers from arXiv."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    # Configure client with delay to respect rate limits (3 seconds between requests)
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,
        num_retries=3
    )

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

Score this paper with a single decimal score from 0.0 to 10.0 (use decimals for precision), considering:
- Innovation: Does it challenge current assumptions or introduce a genuinely new approach?
- Impact: Does it show significant performance improvements or strong real-world potential?
- Methodology: Is the method solid, reproducible and clearly explained?

Return ONLY valid JSON with exactly these keys:
{{"score": <decimal number 0.0-10.0>, "reasoning": "<one sentence explaining the score>"}}

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
# PHASE 2: PDF EXTRACTION AND DETAILED SCORING
# ============================================================================


def download_pdf(url: str, timeout: int = 30) -> bytes:
    """Download PDF from URL and return bytes."""
    try:
        response = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; PaperScope/1.0)'
        })
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error("Failed to download PDF from %s: %s", url, str(e))
        raise


def extract_text_from_pdf(pdf_bytes: bytes, stop_at_references: bool = True) -> str:
    """
    Extract text from PDF bytes.
    Optionally stops at References/Bibliography section.
    """
    if PdfReader is None:
        raise ImportError("pypdf library not installed")
    
    try:
        pdf = PdfReader(BytesIO(pdf_bytes))
        full_text = []
        
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
                
            # Check if we've reached the references section
            if stop_at_references:
                # Look for common references headers
                lines = text.split('\n')
                for i, line in enumerate(lines):
                    line_lower = line.strip().lower()
                    # Common reference section headers
                    if line_lower in ['references', 'bibliography', 'works cited']:
                        # Include text up to this line
                        full_text.append('\n'.join(lines[:i]))
                        logger.info("Stopped at references section on page %d", page_num + 1)
                        return '\n'.join(full_text)
            
            full_text.append(text)
        
        return '\n'.join(full_text)
    
    except Exception as e:
        logger.error("Failed to extract text from PDF: %s", str(e))
        raise


def fetch_and_extract_paper_text(paper: dict) -> str:
    """
    Download PDF and extract text for a paper.
    Returns the extracted text (stopping at references).
    """
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        raise ValueError(f"Paper {paper.get('id')} has no pdf_url")
    
    logger.info("Downloading PDF: %s", paper.get("title", "Unknown")[:50])
    pdf_bytes = download_pdf(pdf_url)
    
    logger.info("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_bytes, stop_at_references=True)
    
    logger.info("Extracted %d characters", len(text))
    return text


def build_detailed_scoring_prompt(paper_text: str, title: str) -> str:
    """Build prompt for detailed 3-criteria scoring based on full PDF."""
    return f"""You are a data science expert curating the most interesting AI/ML research papers for a LinkedIn audience of data scientists and engineers.

IMPORTANT: You are evaluating recently published research from {datetime.now().strftime('%B %Y')}. Your training data may be outdated. Do not penalize papers for mentioning tools, models, or techniques that are newer than your knowledge cutoff. Focus on the research contribution itself.

Score this paper on 3 separate criteria (each 0.0-10.0, use decimals):

1. innovation (weight 50%): How much does this challenge current assumptions or introduce a genuinely new approach? Does it contradict what's currently considered "best practice"?

2. impact (weight 30%): What is the performance improvement? What is the future growth potential for real-world applications?

3. methodology (weight 20%): Is it well-explained with possibility of real application? Is the methodology solid and reproducible?

Return ONLY valid JSON with exactly these keys:
{{"innovation": <0.0-10.0>, "impact": <0.0-10.0>, "methodology": <0.0-10.0>, "reasoning": "<2-3 sentences explaining the scores>"}}

Title: {title}

Full paper:
{paper_text}

JSON:""".strip()


def score_papers_with_pdf(
    papers: list[dict],
    model_id: str = DEFAULT_INFERENCE_PROFILE,
    region: str = "us-east-1",
) -> list[dict]:
    """
    Phase 2: Download PDFs for papers, score them with detailed 3-criteria scoring.
    Returns list of papers with detailed scores and weighted final score.
    """
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    detailed_scores: list[dict] = []

    for i, paper in enumerate(papers, start=1):
        logger.info("\n" + "="*60)
        logger.info("Processing paper %d/%d", i, len(papers))
        logger.info("="*60)
        
        try:
            # Extract PDF text
            paper_text = fetch_and_extract_paper_text(paper)
            
            # Build prompt
            prompt = build_detailed_scoring_prompt(paper_text, paper.get("title", ""))
            
            # Call model with retry logic
            last_err = None
            for attempt in range(1, 6):
                try:
                    actual_text = invoke_bedrock(bedrock, model_id, prompt)
                    data = parse_json_from_model_text(actual_text)
                    
                    # Extract individual scores
                    innovation = float(data.get("innovation", 0))
                    impact = float(data.get("impact", 0))
                    methodology = float(data.get("methodology", 0))
                    reasoning = data.get("reasoning", "")
                    
                    # Calculate weighted score: 50% innovation, 30% impact, 20% methodology
                    weighted_score = (innovation * 0.5) + (impact * 0.3) + (methodology * 0.2)
                    
                    detailed_scores.append({
                        "id": paper.get("id"),
                        "innovation": innovation,
                        "impact": impact,
                        "methodology": methodology,
                        "weighted_score": round(weighted_score, 2),
                        "reasoning": reasoning,
                        "_title": paper.get("title"),
                        "_pdf_url": paper.get("pdf_url"),
                    })
                    
                    logger.info("Scores - Innovation: %.1f, Impact: %.1f, Methodology: %.1f", 
                               innovation, impact, methodology)
                    logger.info("Weighted score: %.2f", weighted_score)
                    break
                    
                except Exception as e:
                    last_err = e
                    logger.warning("Attempt %d failed: %s", attempt, str(e))
                    time.sleep(min(2 ** attempt, 10))
            else:
                logger.error("Failed to score paper after retries: %s", str(last_err))
                detailed_scores.append({
                    "id": paper.get("id"),
                    "innovation": 0,
                    "impact": 0,
                    "methodology": 0,
                    "weighted_score": 0,
                    "reasoning": "",
                    "_title": paper.get("title"),
                    "_error": str(last_err),
                })
        
        except Exception as e:
            logger.error("Failed to process paper: %s", str(e))
            detailed_scores.append({
                "id": paper.get("id"),
                "innovation": 0,
                "impact": 0,
                "methodology": 0,
                "weighted_score": 0,
                "reasoning": "",
                "_title": paper.get("title"),
                "_error": str(e),
            })
        
        time.sleep(0.2)
    
    logger.info("\nDetailed scoring complete: %d papers scored", len(detailed_scores))
    return detailed_scores


def get_top_papers_by_weighted_score(
    papers: list[dict],
    detailed_scores: list[dict],
    top_n: int = 3
) -> list[dict]:
    """
    Select top N papers by weighted score from Phase 2.
    Returns full paper dicts with all scoring details attached.
    """
    papers_by_id = {p["id"]: p for p in papers}
    
    # Sort by weighted_score descending
    sorted_scores = sorted(detailed_scores, key=lambda x: x["weighted_score"], reverse=True)
    
    top = []
    for s in sorted_scores[:top_n]:
        paper = papers_by_id.get(s["id"])
        if paper:
            # Attach all detailed scores to paper
            paper["innovation"] = s["innovation"]
            paper["impact"] = s["impact"]
            paper["methodology"] = s["methodology"]
            paper["weighted_score"] = s["weighted_score"]
            paper["detailed_reasoning"] = s["reasoning"]
            top.append(paper)
    
    return top


# ============================================================================
# PHASE 3: SUMMARY GENERATION
# ============================================================================

def build_summary_prompt(paper_text: str, title: str) -> str:
    """Build prompt for generating accessible summaries of final papers."""
    return f"""You are writing for an audience of data scientists and engineers, including those early in their careers who want to understand cutting-edge AI/ML research.

Write a clear, accessible summary (3-4 paragraphs) of this research paper that explains:
1. What problem does this paper solve and why does it matter?
2. What methodology or approach does the paper use? (Explain the key techniques in plain language)
3. What is the main contribution or innovation?
4. Why should data scientists care? What are the practical implications or potential applications?

Use plain English. Avoid jargon where possible, or explain technical terms when necessary. Focus on making the research understandable and relevant to practitioners.

Title: {title}

Full paper:
{paper_text}

Summary:""".strip()


def generate_summaries_for_papers(
    papers: list[dict],
    model_id: str = DEFAULT_INFERENCE_PROFILE,
    region: str = "us-east-1",
) -> list[dict]:
    """
    Phase 3: Generate accessible summaries for final papers.
    Returns papers with summaries attached.
    """
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    papers_with_summaries = []

    for i, paper in enumerate(papers, start=1):
        logger.info("\n" + "="*60)
        logger.info("Generating summary for paper %d/%d", i, len(papers))
        logger.info("Title: %s", paper.get("title", "")[:80])
        logger.info("="*60)
        
        try:
            # Extract PDF text
            paper_text = fetch_and_extract_paper_text(paper)
            
            # Build summary prompt
            prompt = build_summary_prompt(paper_text, paper.get("title", ""))
            
            # Call model with retry logic
            last_err = None
            for attempt in range(1, 6):
                try:
                    # Note: summaries can be longer, so increase max_tokens
                    response = bedrock.invoke_model(
                        modelId=model_id,
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 1500,  # Longer for summaries
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
                    summary = response_obj.get("content", [{}])[0].get("text", "").strip()
                    
                    # Attach summary to paper
                    paper_copy = paper.copy()
                    paper_copy["summary"] = summary
                    papers_with_summaries.append(paper_copy)
                    
                    logger.info("Summary generated (%d characters)", len(summary))
                    break
                    
                except Exception as e:
                    last_err = e
                    logger.warning("Attempt %d failed: %s", attempt, str(e))
                    time.sleep(min(2 ** attempt, 10))
            else:
                logger.error("Failed to generate summary after retries: %s", str(last_err))
                paper_copy = paper.copy()
                paper_copy["summary"] = ""
                paper_copy["_error"] = str(last_err)
                papers_with_summaries.append(paper_copy)
        
        except Exception as e:
            logger.error("Failed to process paper: %s", str(e))
            paper_copy = paper.copy()
            paper_copy["summary"] = ""
            paper_copy["_error"] = str(e)
            papers_with_summaries.append(paper_copy)
        
        time.sleep(0.2)
    
    logger.info("\nSummary generation complete!")
    return papers_with_summaries


def create_weekly_output(papers_with_summaries: list[dict], week_date: str = None) -> dict:
    """
    Create the final weekly output structure for the static website.
    """
    if week_date is None:
        week_date = datetime.now().strftime("%Y-%m-%d")
    
    weekly_data = {
        "week_date": week_date,
        "papers": []
    }
    
    for paper in papers_with_summaries:
        weekly_data["papers"].append({
            "title": paper.get("title"),
            "summary": paper.get("summary", ""),
            "arxiv_url": f"https://arxiv.org/abs/{paper.get('id')}",
            "arxiv_id": paper.get("id"),
            "scores": {
                "innovation": paper.get("innovation"),
                "impact": paper.get("impact"),
                "methodology": paper.get("methodology"),
                "weighted": paper.get("weighted_score")
            }
        })
    
    return weekly_data


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
    papers_path = Path("data/papers.json")
    if not papers_path.exists():
        logger.info("Fetching papers from arXiv...")
        start = time.time()
        papers = fetch_arxiv_papers()
        logger.info("Fetched %d papers in %.2f seconds", len(papers), time.time() - start)
        save_json(papers, "data/papers.json")
        logger.info("Saved to data/papers.json")
    else:
        logger.info("Papers already exist, loading from data/papers.json")
        papers = load_json("data/papers.json")
        logger.info("Loaded %d papers", len(papers))

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
    top_papers_path = Path("data/top_papers.json")
    if not top_papers_path.exists():
        top_papers = get_top_papers(papers, scores, top_n=10)
        save_json(top_papers, "data/top_papers.json")
        logger.info("Top 10 papers saved to data/top_papers.json")
    else:
        logger.info("Top papers already exist, loading from data/top_papers.json")
        top_papers = load_json("data/top_papers.json")

    logger.info("\nTop 10 papers:")
    for i, p in enumerate(top_papers, 1):
        logger.info("%d. [%.1f] %s", i, p["score"], p["title"])

    # -------------------------------------------------------------------------
    # Step 4: Phase 2 - Detailed scoring with PDF content (top 10 → top 3)
    # -------------------------------------------------------------------------
    detailed_scores_path = Path("data/detailed_scores.json")
    if not detailed_scores_path.exists():
        logger.info("\n" + "="*60)
        logger.info("Phase 2: Detailed scoring with PDF content...")
        logger.info("="*60)
        
        detailed_scores = score_papers_with_pdf(top_papers)
        save_json(detailed_scores, "data/detailed_scores.json")
        logger.info("Saved detailed scores to data/detailed_scores.json")
    else:
        logger.info("Detailed scores already exist, loading from data/detailed_scores.json")
        detailed_scores = load_json("data/detailed_scores.json")

    # -------------------------------------------------------------------------
    # Step 5: Select final top 3 papers
    # -------------------------------------------------------------------------
    final_top_3 = get_top_papers_by_weighted_score(top_papers, detailed_scores, top_n=3)
    save_json(final_top_3, "data/final_top_3.json")
    logger.info("\n" + "="*60)
    logger.info("FINAL TOP 3 PAPERS")
    logger.info("="*60)
    
    for i, p in enumerate(final_top_3, 1):
        logger.info("\n%d. [Weighted: %.2f] %s", i, p["weighted_score"], p["title"])
        logger.info("   Innovation: %.1f | Impact: %.1f | Methodology: %.1f",
                   p["innovation"], p["impact"], p["methodology"])

    # -------------------------------------------------------------------------
    # Step 6: Phase 3 - Generate accessible summaries
    # -------------------------------------------------------------------------
    summaries_path = Path("data/papers_with_summaries.json")
    if not summaries_path.exists():
        logger.info("\n" + "="*60)
        logger.info("Phase 3: Generating accessible summaries...")
        logger.info("="*60)
        
        papers_with_summaries = generate_summaries_for_papers(final_top_3)
        save_json(papers_with_summaries, "data/papers_with_summaries.json")
        logger.info("Saved papers with summaries to data/papers_with_summaries.json")
    else:
        logger.info("Summaries already exist, loading from data/papers_with_summaries.json")
        papers_with_summaries = load_json("data/papers_with_summaries.json")

    # -------------------------------------------------------------------------
    # Step 7: Create weekly output for website
    # -------------------------------------------------------------------------
    week_date = datetime.now().strftime("%Y-%m-%d")
    weekly_output = create_weekly_output(papers_with_summaries, week_date)
    
    # Save to weeks directory
    weeks_dir = Path("data/weeks")
    weeks_dir.mkdir(parents=True, exist_ok=True)
    weekly_file = weeks_dir / f"{week_date}.json"
    save_json(weekly_output, str(weekly_file))
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*60)
    logger.info("Weekly output saved to: %s", weekly_file)
    logger.info("\nFinal papers for week of %s:", week_date)
    for i, paper in enumerate(weekly_output["papers"], 1):
        logger.info("\n%d. %s", i, paper["title"])
        logger.info("   arXiv: %s", paper["arxiv_url"])
        logger.info("   Summary preview: %s...", paper["summary"][:100])