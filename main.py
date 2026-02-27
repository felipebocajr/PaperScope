import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from pathlib import Path

import arxiv
import boto3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

# -------------------------
# Configuration
# -------------------------

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
GPT_OSS_120B_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")

# Topics with search queries
TOPIC_QUERIES = {
    "agentic_ai": {
        "name": "Agentic AI",
        "query": '''(ti:agent OR ti:agentic OR ti:"tool use" OR ti:planning OR 
                     ti:"multi-agent" OR abs:"autonomous agent" OR abs:"tool calling" OR 
                     abs:"action space")''',
        "max_papers": 200
    },
    "reinforcement_learning": {
        "name": "Reinforcement Learning",
        "query": '''(ti:"reinforcement learning" OR ti:RLHF OR ti:PPO OR ti:DPO OR 
                     ti:"policy optimization" OR abs:"reward model" OR abs:"Q-learning" OR
                     abs:"actor-critic")''',
        "max_papers": 200
    },
    "llms_applications": {
        "name": "LLMs & Applications",
        "query": '''(ti:LLM OR ti:"language model" OR ti:"fine-tuning" OR ti:RAG OR 
                     ti:"prompt engineering" OR abs:"retrieval augmented" OR 
                     abs:"instruction tuning" OR abs:"in-context learning")''',
        "max_papers": 200
    },
    "computer_vision": {
        "name": "Computer Vision",
        "query": '''((ti:vision OR ti:image OR ti:detection OR ti:segmentation OR 
                      ti:"object detection" OR abs:"convolutional" OR abs:"visual recognition")
                     AND NOT (ti:language OR ti:multimodal OR ti:"vision-language" OR 
                              abs:"vision-language" OR abs:"cross-modal"))''',
        "max_papers": 200
    },
    "multimodal": {
        "name": "Multimodal",
        "query": '''(ti:multimodal OR ti:"vision-language" OR ti:CLIP OR ti:"cross-modal" OR 
                     abs:"vision and language" OR abs:"audio-visual" OR abs:"video-language")''',
        "max_papers": 200
    }
}

# Keep old TOPICS dict for compatibility
TOPICS = {k: v["name"] for k, v in TOPIC_QUERIES.items()}
TOPIC_DISPLAY_NAMES = TOPICS

# Pipeline configuration
ARXIV_QUERY = os.getenv("ARXIV_QUERY", "cat:cs.AI")
MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "1000"))
DAYS_LOOKBACK = int(os.getenv("ARXIV_DAYS_LOOKBACK", "7"))
TOP_K_PER_TOPIC = int(os.getenv("TOP_K_PER_TOPIC", "5"))

# Rate limiting
ARXIV_DELAY_SEC = float(os.getenv("ARXIV_DELAY_SEC", "10")) 
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))

# Directories
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
OUT_DIR = Path(os.getenv("OUT_DIR", "./out"))
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# LLM parameters
PHASE1_MAX_TOKENS = 400
PHASE1_TEMPERATURE = 0.0

# Massively increased tokens to prevent "Unterminated string" errors during batch JSON generation
PHASE2_MAX_TOKENS = 2500 
PHASE2_TEMPERATURE = 0.0

PHASE3_MAX_TOKENS = 900
PHASE3_TEMPERATURE = 0.5
PHASE3_TOP_P = 0.9

# Regex helpers
RE_LEADING_REASONING = re.compile(r"^\s*<reasoning>.*?</reasoning>\s*", re.DOTALL)
RE_REFS_LINE = re.compile(r"^\s*(references|bibliography|works\s+cited)\b", re.IGNORECASE)


# -------------------------
# Pydantic Schemas
# -------------------------

class PaperEvaluation(BaseModel):
    paper_number: int = Field(description="The sequential number of the paper in the batch.")
    reasoning: str = Field(description="2-3 sentences evaluating the paper against the rubric before scoring.")
    innovation: float = Field(ge=0.0, le=10.0, description="Score for challenging assumptions or introducing new approaches.")
    impact: float = Field(ge=0.0, le=10.0, description="Score for performance improvements and real-world potential.")
    methodology: float = Field(ge=0.0, le=10.0, description="Score for reproducibility and solid approach.")


# -------------------------
# Data structures
# -------------------------

@dataclass
class Paper:
    title: str
    authors: List[str]
    published: str
    abstract: str
    pdf_url: str
    html_url: str
    entry_id: str
    arxiv_id: str
    
    # Phase 1: Multi-topic scores
    topic_scores: Dict[str, float] = None
    assigned_topic: str = ""
    
    # Phase 2: Detailed scores
    innovation: float = 0.0
    impact: float = 0.0
    methodology: float = 0.0
    weighted_score: float = 0.0
    detailed_reasoning: str = ""
    
    # Phase 3: Summary
    final_summary: str = ""
    
    def __post_init__(self):
        if self.topic_scores is None:
            self.topic_scores = {}


# -------------------------
# Bedrock invocation (Restored to boto3)
# -------------------------

def invoke_bedrock(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    """Call Bedrock and return text response."""
    
    # OpenAI GPT-OSS models
    if model_id.startswith("openai."):
        native_request = {
            "messages": [
                {"role": "system", "content": "Return only what the user asks for."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(native_request),
        )
        
        raw = response.get("body")
        if raw is None:
            raise RuntimeError("Empty response body from Bedrock")
        
        payload = raw.read()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="ignore")
        
        response_obj = json.loads(payload)
        actual_text = (
            response_obj.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        )
        actual_text = actual_text.strip()
        actual_text = RE_LEADING_REASONING.sub("", actual_text).strip()
        return actual_text
    
    # Anthropic models
    anthropic_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }
    
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(anthropic_body),
    )
    
    raw = response.get("body")
    if raw is None:
        raise RuntimeError("Empty response body from Bedrock")
    
    payload = raw.read()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="ignore")
    
    response_obj = json.loads(payload)
    actual_text = response_obj.get("content", [{}])[0].get("text", "")
    return actual_text.strip()


# -------------------------
# Prompts
# -------------------------

def build_comparative_batch_scoring_prompt(papers_batch: List[Paper]) -> str:
    """Score a batch of papers comparatively with calibration."""
    papers_text = ""
    for i, paper in enumerate(papers_batch, 1):
        papers_text += f"\n{'='*60}\nPAPER {i}:\nTitle: {paper.title}\n\nAbstract:\n{paper.abstract}\n"
    
    # FIXED: Python 3.12 datetime warning resolved using timezone.utc
    current_month_year = datetime.now(timezone.utc).strftime('%B %Y')
    
    return f"""You are evaluating research papers for an AI/ML digest.

IMPORTANT: You are evaluating recently published research from {current_month_year}. Do not penalize papers for mentioning newer tools or models.

You will score {len(papers_batch)} papers on 3 criteria (each 0.0-10.0, use decimals):

1. innovation (weight 50%): Does it challenge assumptions or introduce genuinely new approaches?
2. impact (weight 30%): Performance improvements? Real-world application potential?
3. methodology (weight 20%): Well-explained? Reproducible? Solid approach?

<calibration>
Use the FULL 0-10 scale. Here's what each range means:

9.0-10.0: Groundbreaking work that will likely be cited for years (rare - maybe 1 in 100 papers)
7.5-8.9: Strong contribution with clear novelty and solid execution (top 10-15% of papers)
5.5-7.4: Decent work but incremental improvement or limited scope (typical good paper)
3.0-5.4: Marginal contribution or significant methodological concerns
0.0-2.9: Fundamentally flawed or trivial work (rare)

Most papers should fall in the 4.0-8.5 range. Don't cluster everything around 8.0!
</calibration>

<comparative_evaluation>
Compare these {len(papers_batch)} papers to each other AND to typical arXiv submissions:
- Which papers stand out as exceptional within this batch?
- Which are solid but incremental?
- Which have significant limitations?
- Use the full score range - if all papers seem similar quality, you're not evaluating critically enough
</comparative_evaluation>

CRITICAL JSON RULES:
1. Return ONLY a valid JSON array. Do not include markdown blocks.
2. DO NOT use double quotes (") inside the reasoning text. Use single quotes (') instead.
3. DO NOT use newlines inside the reasoning text.

Return ONLY valid JSON array with {len(papers_batch)} objects:
[
  {{
    "paper_number": 1,
    "reasoning": "<2-3 sentences of evaluation BEFORE scoring>",
    "innovation": <0.0-10.0>,
    "impact": <0.0-10.0>,
    "methodology": <0.0-10.0>
  }}
]

{papers_text}

JSON array:""".strip()


def build_summary_prompt(full_text: str, title: str) -> str:
    """Phase 3: Generate accessible summary from full HTML."""
    return f"""You are an expert technical writer creating a weekly AI/ML research digest for practitioners (data scientists, ML engineers, technical leaders).

Your task: Write a concise, engaging summary of the research paper below.

<audience>
Readers keep up with research but don't read full papers. They understand ML fundamentals (neural networks, training data, gradients) but appreciate clear explanations of specialized techniques.
</audience>

<style_requirements>
- Professional but conversational (like explaining to a colleague)
- Use metaphors only when they add clarity, not for decoration
- No headings, numbered lists, or bullet points
- Vary sentence structure and length; use active voice
- When introducing specialized terms, briefly clarify in plain language
</style_requirements>

<content_structure>
Write 3-4 short paragraphs that flow naturally:
1. Start with a compelling hook (what's interesting or surprising?)
2. Explain the core approach and what changed versus typical methods
3. Include 1-2 concrete details (metrics, scale, dataset, or key results)
4. Close with practical implications (why should practitioners care?)
</content_structure>

<tone_guidelines>
- More grounded than creative writing, less formal than academic prose
- Think "high-quality technical blog post" not "literature review"
- Vary opening hooks—avoid repeating "Imagine..." across papers
- Be direct and clear over clever
</tone_guidelines>

<constraints>
- Target length: 220-260 words
- Must fit naturally into a curated weekly digest
</constraints>

<paper>
Title: {title}

Content:
{full_text}
</paper>

Write the summary now (output only the summary text, no preamble):""".strip()


# -------------------------
# arXiv fetching with topic queries
# -------------------------

def fetch_papers_for_topic(topic_key: str, days_back: int = DAYS_LOOKBACK) -> List[Paper]:
    """Fetch papers for a specific topic using targeted search query."""
    topic_config = TOPIC_QUERIES[topic_key]
    query = topic_config["query"]
    max_papers = topic_config["max_papers"]
    topic_name = topic_config["name"]
    
    print(f"\nFetching {topic_name} papers...")
    print(f"  Query: {query[:80]}...")
    
    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    client = arxiv.Client(
        page_size=100,
        delay_seconds=ARXIV_DELAY_SEC,
        num_retries=3
    )
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    papers = []
    
    for result in client.results(search):
        if result.published and result.published >= cutoff:
            arxiv_id = result.get_short_id()
            html_url = f"https://arxiv.org/html/{arxiv_id}"
            
            paper = Paper(
                title=result.title,
                authors=[a.name for a in result.authors],
                published=result.published.strftime("%Y-%m-%d"),
                abstract=result.summary,
                pdf_url=result.pdf_url,
                html_url=html_url,
                entry_id=result.entry_id,
                arxiv_id=arxiv_id,
            )
            # Pre-assign topic since we fetched it specifically for this topic
            paper.assigned_topic = topic_key
            papers.append(paper)
    
    print(f"  ✓ Fetched {len(papers)} papers")
    return papers


def fetch_all_papers_by_topic() -> Dict[str, List[Paper]]:
    """Fetch papers for all topics using targeted queries."""
    print("="*60)
    print("Fetching papers by topic using targeted queries")
    print("="*60)
    
    papers_by_topic = {}
    
    for topic_key in TOPIC_QUERIES.keys():
        papers = fetch_papers_for_topic(topic_key)
        papers_by_topic[topic_key] = papers
        print("  Waiting 5 seconds to respect arXiv rate limits...")
        time.sleep(5)  # Polite delay between topic queries
    
    total = sum(len(papers) for papers in papers_by_topic.values())
    print(f"\n✓ Total papers fetched: {total}")
    
    return papers_by_topic


# -------------------------
# HTML extraction
# -------------------------

def download_and_extract_html(paper: Paper) -> str:
    """Download HTML version of paper and extract main content."""
    try:
        print(f"  Downloading HTML: {paper.title[:60]}...")
        response = requests.get(paper.html_url, timeout=HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract main content (arXiv HTML has specific structure)
        # Try to find article body or main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if not main_content:
            print("  Warning: Could not find main content, using full HTML")
            text = soup.get_text(separator='\n', strip=True)
        else:
            text = main_content.get_text(separator='\n', strip=True)
        
        # Try to stop at references
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if RE_REFS_LINE.match(line.strip()):
                print(f"  Stopped at references (line {idx})")
                text = '\n'.join(lines[:idx])
                break
        
        print(f"  Extracted {len(text)} characters")
        return text
        
    except Exception as e:
        print(f"  Error extracting HTML: {e}")
        return ""


# -------------------------
# Phase 2: Detailed scoring
# -------------------------

def score_papers_detailed(papers_by_topic: Dict[str, List[Paper]], top_k: int = TOP_K_PER_TOPIC, model_id: str = GPT_OSS_120B_MODEL_ID) -> Dict[str, List[Paper]]:
    """Score papers in detail using comparative batch evaluation, then pick top K per topic."""
    print(f"\nDetailed comparative scoring (batches of 10, top {top_k} per topic)...")
    print(f"Using model: {model_id}")
    
    # Restored boto3 client
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    scored_by_topic = {}
    
    BATCH_SIZE = 10
    
    for topic_key, topic_papers in papers_by_topic.items():
        topic_name = TOPIC_QUERIES[topic_key]["name"]
        
        if not topic_papers:
            print(f"\n{topic_name}: No papers")
            scored_by_topic[topic_key] = []
            continue
        
        print(f"\n{topic_name}: Scoring {len(topic_papers)} papers in batches of {BATCH_SIZE}...")
        
        # Process papers in batches
        for batch_start in range(0, len(topic_papers), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(topic_papers))
            batch = topic_papers[batch_start:batch_end]
            
            print(f"  Batch {batch_start//BATCH_SIZE + 1}: Papers {batch_start+1}-{batch_end}...")
            
            try:
                prompt = build_comparative_batch_scoring_prompt(batch)
                
                response_text = invoke_bedrock(
                    bedrock,
                    model_id,
                    prompt,
                    max_tokens=PHASE2_MAX_TOKENS,
                    temperature=0.2,
                )
                
                # Robust JSON array extraction using regex
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
                if not json_match:
                    raise ValueError("Could not locate a valid JSON array in the response.")
                
                clean_json_str = json_match.group(0)
                raw_json = json.loads(clean_json_str)
                
                # Hybrid Pydantic Validation
                try:
                    evaluations = [PaperEvaluation.model_validate(item) for item in raw_json]
                except ValidationError as e:
                    print(f"  Pydantic Validation Error: {e}")
                    raise e
                
                # Assign scores to papers
                for score_obj in evaluations:
                    paper_idx = score_obj.paper_number - 1
                    if 0 <= paper_idx < len(batch):
                        paper = batch[paper_idx]
                        paper.innovation = score_obj.innovation
                        paper.impact = score_obj.impact
                        paper.methodology = score_obj.methodology
                        paper.detailed_reasoning = score_obj.reasoning
                        
                        # Calculate weighted score
                        paper.weighted_score = (
                            paper.innovation * 0.5 +
                            paper.impact * 0.3 +
                            paper.methodology * 0.2
                        )
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"  Error scoring batch: {e}")
                # Assign default scores to batch
                for paper in batch:
                    paper.weighted_score = 0.0
        
        # Sort by weighted score and take top K
        topic_papers.sort(key=lambda p: p.weighted_score, reverse=True)
        top_papers = topic_papers[:top_k]
        
        print(f"\n  Top {len(top_papers)} papers:")
        for i, paper in enumerate(top_papers, 1):
            print(f"    {i}. [{paper.weighted_score:.2f}] (I:{paper.innovation:.1f} M:{paper.impact:.1f} Me:{paper.methodology:.1f}) {paper.title[:50]}")
        
        scored_by_topic[topic_key] = top_papers
    
    print("\n✓ Detailed scoring complete")
    return scored_by_topic


# -------------------------
# Phase 3: Generate summaries
# -------------------------

def generate_summaries_for_winners(papers_by_topic: Dict[str, List[Paper]], model_id: str = GPT_OSS_120B_MODEL_ID) -> Dict[str, Paper]:
    """Generate summaries for the best paper in each topic."""
    print(f"\nGenerating summaries (1 per topic)...")
    print(f"Using model: {model_id}")
    
    # Restored boto3 client
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    winners = {}
    
    for topic_key, topic_papers in papers_by_topic.items():
        if not topic_papers:
            continue
        
        # Pick best paper in this topic
        best_paper = max(topic_papers, key=lambda p: p.weighted_score)
        topic_name = TOPIC_DISPLAY_NAMES[topic_key]
        
        print(f"\n{topic_name}:")
        print(f"  Winner: {best_paper.title[:70]}")
        print(f"  Score: {best_paper.weighted_score:.2f}")
        
        try:
            # Download HTML content
            html_content = download_and_extract_html(best_paper)
            
            if not html_content:
                print("  Warning: Using abstract as fallback")
                html_content = best_paper.abstract
            
            # Generate summary
            prompt = build_summary_prompt(html_content, best_paper.title)
            
            summary = invoke_bedrock(
                bedrock,
                model_id,
                prompt,
                max_tokens=PHASE3_MAX_TOKENS,
                temperature=PHASE3_TEMPERATURE,
                top_p=PHASE3_TOP_P,
            )
            
            best_paper.final_summary = summary
            print(f"  Summary: {len(summary)} chars")
            
            winners[topic_key] = best_paper
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  Error: {e}")
            best_paper.final_summary = ""
            winners[topic_key] = best_paper
    
    print("✓ Phase 3 complete")
    return winners


# -------------------------
# Output
# -------------------------

def create_weekly_output(winners: Dict[str, Paper], week_date: str) -> dict:
    """Create final weekly JSON output."""
    output = {
        "week_date": week_date,
        # FIXED: Python 3.12 datetime warning resolved using timezone.utc
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": GPT_OSS_120B_MODEL_ID,
        "topics": {}
    }
    
    for topic_key, paper in winners.items():
        output["topics"][topic_key] = {
            "topic_name": TOPIC_QUERIES[topic_key]["name"],
            "paper": {
                "title": paper.title,
                "authors": paper.authors,
                "published": paper.published,
                "arxiv_url": f"https://arxiv.org/abs/{paper.arxiv_id}",
                "arxiv_id": paper.arxiv_id,
                "summary": paper.final_summary,
                "scores": {
                    "innovation": paper.innovation,
                    "impact": paper.impact,
                    "methodology": paper.methodology,
                    "weighted": paper.weighted_score,
                }
            }
        }
    
    return output


# -------------------------
# Main
# -------------------------

def main():
    print("="*60)
    print("AI Research Digest - Topic Query Pipeline")
    print("="*60)
    
    # FIXED: Using timezone.utc to be safe
    week_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Fetch papers by topic using targeted queries
    papers_by_topic = fetch_all_papers_by_topic()
    
    # Score papers in detail and pick top K per topic
    scored_papers = score_papers_detailed(papers_by_topic)
    
    # Generate summaries for winners
    winners = generate_summaries_for_winners(scored_papers)
    
    # Create output
    output = create_weekly_output(winners, week_date)
    
    # Save
    out_file = OUT_DIR / f"{week_date}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Output: {out_file}")
    print(f"\nWeekly digest for {week_date}:")
    for topic_key, paper in winners.items():
        topic_name = TOPIC_QUERIES[topic_key]["name"]
        print(f"\n{topic_name}:")
        print(f"  {paper.title}")
        print(f"  Score: {paper.weighted_score:.2f}")


if __name__ == "__main__":
    main()