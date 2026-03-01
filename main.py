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

# Optional: S3 Upload
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "")

# Topics with search queries
TOPIC_QUERIES = {
    "agentic_ai": {
        "name": "Agentic AI",
        "query": '''(ti:agent OR ti:agentic OR ti:"tool use" OR ti:planning OR 
                     ti:"multi-agent" OR abs:"autonomous agent" OR abs:"tool calling" OR 
                     abs:"action space")''',
        "max_papers": 300
    },
    "reinforcement_learning": {
        "name": "Reinforcement Learning",
        "query": '''(ti:"reinforcement learning" OR ti:RLHF OR ti:PPO OR ti:DPO OR 
                     ti:"policy optimization" OR abs:"reward model" OR abs:"Q-learning" OR
                     abs:"actor-critic")''',
        "max_papers": 300
    },
    "llms_applications": {
        "name": "LLMs & Applications",
        "query": '''(ti:LLM OR ti:"language model" OR ti:"fine-tuning" OR ti:RAG OR 
                     ti:"prompt engineering" OR abs:"retrieval augmented" OR 
                     abs:"instruction tuning" OR abs:"in-context learning")''',
        "max_papers": 300
    },
    "computer_vision": {
        "name": "Computer Vision",
        "query": '''((ti:vision OR ti:image OR ti:detection OR ti:segmentation OR 
                      ti:"object detection" OR abs:"convolutional" OR abs:"visual recognition")
                     AND NOT (ti:language OR ti:multimodal OR ti:"vision-language" OR 
                              abs:"vision-language" OR abs:"cross-modal"))''',
        "max_papers": 300
    },
    "multimodal": {
        "name": "Multimodal",
        "query": '''(ti:multimodal OR ti:"vision-language" OR ti:CLIP OR ti:"cross-modal" OR 
                     abs:"vision and language" OR abs:"audio-visual" OR abs:"video-language")''',
        "max_papers": 300
    }
}

TOPICS = {k: v["name"] for k, v in TOPIC_QUERIES.items()}
TOPIC_DISPLAY_NAMES = TOPICS

# FIXED: Strict CS Categories so we don't grab Physics/Math papers by accident
ARXIV_QUERY = os.getenv("ARXIV_QUERY", "(cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL)")
MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "1000"))
DAYS_LOOKBACK = int(os.getenv("ARXIV_DAYS_LOOKBACK", "7"))
TOP_K_PER_TOPIC = int(os.getenv("TOP_K_PER_TOPIC", "5"))

# Rate limiting
ARXIV_DELAY_SEC = 5
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))

# Directories
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
OUT_DIR = Path(os.getenv("OUT_DIR", "./out"))
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# LLM parameters
PHASE1_MAX_TOKENS = 400
PHASE1_TEMPERATURE = 0.0

PHASE2_MAX_TOKENS = 3500 
PHASE2_TEMPERATURE = 0.5 

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
    
    assigned_topic: str = ""
    
    # Phase 2: Detailed scores
    innovation: float = 0.0
    impact: float = 0.0
    methodology: float = 0.0
    weighted_score: float = 0.0
    detailed_reasoning: str = ""
    
    # Phase 3: Summary
    final_summary: str = ""


# -------------------------
# Bedrock invocation
# -------------------------

def invoke_bedrock(bedrock_client, model_id: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0) -> str:
    """Call Bedrock and return text response."""
    if model_id.startswith("openai."):
        native_request = {
            "messages": [
                {"role": "system", "content": "Return only what the user asks for."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(native_request),
        )
        raw = response.get("body")
        payload = raw.read().decode("utf-8", errors="ignore")
        response_obj = json.loads(payload)
        actual_text = response_obj.get("choices", [{}])[0].get("message", {}).get("content", "")
        return RE_LEADING_REASONING.sub("", actual_text).strip()
    
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
    payload = raw.read().decode("utf-8", errors="ignore")
    response_obj = json.loads(payload)
    return response_obj.get("content", [{}])[0].get("text", "").strip()


# -------------------------
# Prompts
# -------------------------

def build_comparative_batch_scoring_prompt(papers_batch: List[Paper]) -> str:
    """PASS 1: Score a batch of papers quickly to filter out the noise."""
    papers_text = ""
    for i, paper in enumerate(papers_batch, 1):
        papers_text += f"\n{'='*60}\nPAPER {i}:\nTitle: {paper.title}\n\nAbstract:\n{paper.abstract}\n"
    
    num_papers = len(papers_batch)
    return f"""You are an expert AI/ML research evaluator. Evaluate this batch of papers.

SCORING CRITERIA (0.0-10.0 scale):
- Innovation: Does the paper challenge assumptions or introduce genuinely novel approaches?
- Impact: Performance improvements? Real-world application potential?
- Methodology: Well-explained? Reproducible? Solid technical approach?

OUTPUT_FORMAT & CRITICAL RULES:
1. DO NOT use double quotes (") inside the reasoning text. Use single quotepss (') instead.
2. DO NOT use newlines inside the reasoning text.

Return ONLY a valid JSON array with exactly {num_papers} objects in this format:
[
  {{
    "paper_number": 1,
    "reasoning": "[Your comparative analysis for this paper, 2-3 sentences]",
    "innovation": <score 0.0-10.0>,
    "impact": <score 0.0-10.0>,
    "methodology": <score 0.0-10.0>
  }}
]

PAPERS TO EVALUATE:
{papers_text}

JSON array:""".strip()


def build_finals_scoring_prompt(papers_batch: List[Paper]) -> str:
    """PASS 2: The Finals. Strictly rank the top papers against each other to fix the baseline."""
    papers_text = ""
    for i, paper in enumerate(papers_batch, 1):
        papers_text += f"\n{'='*60}\nPAPER {i}:\nTitle: {paper.title}\n\nAbstract:\n{paper.abstract}\n"
    
    num_papers = len(papers_batch)
    return f"""You are an expert AI/ML research evaluator judging the "Finals" for a weekly digest.
These {num_papers} papers are the absolute best papers of the week. 

YOUR TASK:
Read all {num_papers} abstracts and rank them STRICTLY against each other. 
You must establish a global baseline: the absolute best paper in this group MUST get the highest score, and the relatively weakest in this group must get the lowest.
Force a spread in the scores. Do NOT give multiple papers the exact same scores.

SCORING CRITERIA (0.0-10.0 scale, use decimals):
- Innovation: Novelty and paradigm-shifting ideas.
- Impact: Real-world utility and performance gains.
- Methodology: Rigor and reproducibility.

OUTPUT_FORMAT & CRITICAL RULES:
1. NO TIES ALLOWED: Force a strict ranking/spread of scores.
2. DO NOT use double quotes (") inside the reasoning text. Use single quotes (') instead.
3. DO NOT use newlines inside the reasoning text.

Return ONLY a valid JSON array with exactly {num_papers} objects in this format:
[
  {{
    "paper_number": 1,
    "reasoning": "[Your strict comparative ranking analysis, 2-3 sentences]",
    "innovation": <score 0.0-10.0>,
    "impact": <score 0.0-10.0>,
    "methodology": <score 0.0-10.0>
  }}
]

PAPERS TO EVALUATE:
{papers_text}

JSON array:""".strip()


def build_summary_prompt(full_text: str, title: str) -> str:
    """Phase 3: Generate accessible summary from full HTML."""
    return f"""You are an expert technical writer creating research summaries for a diverse AI/ML audience. Your summaries appear in a weekly digest read by practitioners ranging from early-career data scientists to technical managers.

<audience>
Your readers span from junior practitioners to technical leaders:
- Early-career data scientists and ML engineers learning the field
- Experienced practitioners keeping up with research
- Technical managers making technology decisions
All understand ML fundamentals, but depth of expertise varies. Balance technical precision with clarity.
</audience>

<writing_principles>
- Be direct and specific - avoid vague openings like "This paper addresses..."
- Use technical terminology when necessary, but briefly explain specialized concepts inline
- Assume familiarity with core ML (neural networks, training, inference) but not niche subfields
- Write with confidence - state what the work accomplishes, not what it "attempts"
- Vary your narrative structure to match each paper's contribution
- Make technical depth accessible without oversimplifying
</writing_principles>

<content_requirements>
Your summary should naturally incorporate:
- What makes this work significant (lead with this if compelling)
- The core technical approach - explain key innovations clearly
- Concrete results: metrics, scale, comparisons, or theoretical guarantees
- Practical relevance: why this matters for building systems, choosing methods, or understanding the field

The order and emphasis should fit the paper - don't force a formula.
</content_requirements>

<style_guidance>
- Professional and precise, but not overly academic
- More akin to a technical blog (Distill, The Gradient) than a research abstract
- When using specialized terms, provide brief context (e.g., "KL divergence, a measure of distribution difference")
- Avoid overused phrases: "game-changer," "revolutionary," "unlock," "leverage"
- Avoid formulaic transitions: "The authors demonstrate," "Results show that"
- Use active voice and varied sentence structure
- Be measured about claims - don't amplify hype, but acknowledge genuine advances
</style_guidance>

<constraints>
- Length: 220-260 words
- Format: 3-4 paragraphs with natural flow
- No headings, bullet points, or numbered lists
- Vary opening hooks - avoid repetitive patterns
</constraints>

<reference_example>
Here's an example of the tone and depth to aim for:

"It's startling how a single, cleverly crafted sentence can hijack an entire AI assistant, turning a helpful planner loop into a conduit for unsafe tool calls. That hidden attack surface has been a blind spot ever since developers started stitching together LLM prompts, code execution, and external APIs into autonomous agents.

The authors answer this gap with the LLMbda calculus, an untyped call-by-value lambda calculus—essentially a minimal programming language that evaluates expressions by substituting values—augmented by dynamic information-flow control, which tracks how data moves through the system to enforce security policies..."

Note: Technical terms are used but explained inline. Sophisticated but accessible.
</reference_example>

Title: {title}

Full paper content:
{full_text}

Write the summary now (output only the summary text):""".strip()



# -------------------------
# arXiv fetching with topic queries
# -------------------------

def fetch_papers_for_topic(topic_key: str, days_back: int = DAYS_LOOKBACK) -> List[Paper]:
    """Fetch papers for a specific topic using targeted search query."""
    topic_config = TOPIC_QUERIES[topic_key]
    base_query = topic_config["query"]
    max_papers = topic_config["max_papers"]
    topic_name = topic_config["name"]
    
    # Combined category filter + keyword query
    full_query = f"{ARXIV_QUERY} AND {base_query}"
    
    print(f"\nFetching {topic_name} papers...")
    
    search = arxiv.Search(
        query=full_query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    client = arxiv.Client(page_size=max_papers, delay_seconds=ARXIV_DELAY_SEC, num_retries=3)
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
            paper.assigned_topic = topic_key
            papers.append(paper)
    
    print(f"  ✓ Fetched {len(papers)} papers")
    return papers


def fetch_all_papers_by_topic() -> Dict[str, List[Paper]]:
    print("="*60)
    print("Fetching papers by topic using targeted queries")
    print("="*60)
    
    papers_by_topic = {}
    for topic_key in TOPIC_QUERIES.keys():
        papers = fetch_papers_for_topic(topic_key)
        papers_by_topic[topic_key] = papers
        time.sleep(5)  # Polite delay
    
    return papers_by_topic


# -------------------------
# HTML extraction
# -------------------------

def download_and_extract_html(paper: Paper) -> str:
    try:
        response = requests.get(paper.html_url, timeout=HTTP_TIMEOUT_SEC)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
        
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            if RE_REFS_LINE.match(line.strip()):
                text = '\n'.join(lines[:idx])
                break
        return text
    except Exception as e:
        return ""


# -------------------------
# Phase 2: Tournament Scoring
# -------------------------

def score_papers_detailed(papers_by_topic: Dict[str, List[Paper]], top_k: int = TOP_K_PER_TOPIC, model_id: str = GPT_OSS_120B_MODEL_ID) -> Dict[str, List[Paper]]:
    """Two-Pass Tournament Scoring: Qualifiers -> Finals"""
    print(f"\nStarting Two-Pass Tournament Scoring...")
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    scored_by_topic = {}
    
    BATCH_SIZE = 5
    QUALIFIER_TOP_K = 10  # We send the Top 10 to the Finals
    
    for topic_key, topic_papers in papers_by_topic.items():
        topic_name = TOPIC_QUERIES[topic_key]["name"]
        
        if not topic_papers:
            scored_by_topic[topic_key] = []
            continue
            
        print(f"\n{topic_name} [QUALIFIERS]: Evaluating {len(topic_papers)} papers...")
        
        # --- PASS 1: QUALIFIERS ---
        for batch_start in range(0, len(topic_papers), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(topic_papers))
            batch = topic_papers[batch_start:batch_end]
            
            try:
                prompt = build_comparative_batch_scoring_prompt(batch)
                response_text = invoke_bedrock(bedrock, model_id, prompt, max_tokens=PHASE2_MAX_TOKENS, temperature=PHASE2_TEMPERATURE)
                
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
                raw_json = json.loads(json_match.group(0))
                evaluations = [PaperEvaluation.model_validate(item) for item in raw_json]
                
                for score_obj in evaluations:
                    paper_idx = score_obj.paper_number - 1
                    if 0 <= paper_idx < len(batch):
                        paper = batch[paper_idx]
                        paper.innovation = score_obj.innovation
                        paper.impact = score_obj.impact
                        paper.methodology = score_obj.methodology
                        paper.detailed_reasoning = score_obj.reasoning
                        paper.weighted_score = (paper.innovation * 0.5 + paper.impact * 0.3 + paper.methodology * 0.2)
            except Exception as e:
                for paper in batch: paper.weighted_score = 0.0
        
        # Sort and pick Finalists
        topic_papers.sort(key=lambda p: p.weighted_score, reverse=True)
        finalists = topic_papers[:QUALIFIER_TOP_K]
        
        # --- PASS 2: THE FINALS ---
        print(f"  🏆 [THE FINALS] Calibrating the Top {len(finalists)} papers against each other...")
        try:
            prompt = build_finals_scoring_prompt(finalists)
            # Temperature lowered to 0.3 for strict, analytical ranking
            response_text = invoke_bedrock(bedrock, model_id, prompt, max_tokens=PHASE2_MAX_TOKENS, temperature=0.3)
            
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            raw_json = json.loads(json_match.group(0))
            evaluations = [PaperEvaluation.model_validate(item) for item in raw_json]
            
            for score_obj in evaluations:
                paper_idx = score_obj.paper_number - 1
                if 0 <= paper_idx < len(finalists):
                    paper = finalists[paper_idx]
                    paper.innovation = score_obj.innovation
                    paper.impact = score_obj.impact
                    paper.methodology = score_obj.methodology
                    paper.detailed_reasoning = score_obj.reasoning
                    paper.weighted_score = (paper.innovation * 0.5 + paper.impact * 0.3 + paper.methodology * 0.2)
        except Exception as e:
            print(f"  Error in Finals: {e} - Falling back to Qualifier scores.")
            
        # Final sort after calibration
        finalists.sort(key=lambda p: p.weighted_score, reverse=True)
        top_papers = finalists[:top_k]
        
        print(f"\n  Final Top {len(top_papers)} papers:")
        for i, paper in enumerate(top_papers, 1):
            print(f"    {i}. [{paper.weighted_score:.2f}] (I:{paper.innovation:.1f} M:{paper.impact:.1f} Me:{paper.methodology:.1f}) {paper.title[:50]}")
        
        scored_by_topic[topic_key] = top_papers
    
    return scored_by_topic


# -------------------------
# Phase 3: Generate summaries
# -------------------------

def generate_summaries_for_winners(papers_by_topic: Dict[str, List[Paper]], model_id: str = GPT_OSS_120B_MODEL_ID) -> Dict[str, Paper]:
    print(f"\nGenerating summaries (1 per topic)...")
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    winners = {}
    
    for topic_key, topic_papers in papers_by_topic.items():
        if not topic_papers: continue
        
        best_paper = max(topic_papers, key=lambda p: p.weighted_score)
        print(f"\n{TOPIC_DISPLAY_NAMES[topic_key]}:\n  Winner: {best_paper.title[:70]}")
        
        try:
            html_content = download_and_extract_html(best_paper)
            if not html_content: html_content = best_paper.abstract
            
            prompt = build_summary_prompt(html_content, best_paper.title)
            summary = invoke_bedrock(bedrock, model_id, prompt, max_tokens=PHASE3_MAX_TOKENS, temperature=PHASE3_TEMPERATURE, top_p=PHASE3_TOP_P)
            best_paper.final_summary = summary
            winners[topic_key] = best_paper
        except Exception as e:
            best_paper.final_summary = ""
            winners[topic_key] = best_paper
    
    return winners


# -------------------------
# S3 Upload
# -------------------------

def upload_to_s3(file_path: Path, bucket: str, prefix: str = ""):
    if not bucket: return
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    s3_key = f"{prefix.strip('/')}/{file_path.name}" if prefix else file_path.name
        
    print(f"\nUploading to S3: s3://{bucket}/{s3_key}...")
    try:
        s3_client.upload_file(str(file_path), bucket, s3_key)
        print("  ✓ Upload successful")
    except Exception as e:
        print(f"  ✗ Error uploading to S3: {e}")

def update_weeks_index(week_date: str, bucket: str, out_dir: Path):
    weeks_file = out_dir / "weeks.json"
    weeks_list = []
    
    if bucket:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        try:
            response = s3_client.get_object(Bucket=bucket, Key="weeks.json")
            weeks_list = json.loads(response['Body'].read().decode('utf-8'))
        except Exception:
            if weeks_file.exists():
                with open(weeks_file, 'r', encoding='utf-8') as f: weeks_list = json.load(f)
    else:
        if weeks_file.exists():
            with open(weeks_file, 'r', encoding='utf-8') as f: weeks_list = json.load(f)
    
    if week_date not in weeks_list: weeks_list.append(week_date)
    weeks_list.sort(reverse=True)
    
    with open(weeks_file, 'w', encoding='utf-8') as f: json.dump(weeks_list, f, indent=2)
    if bucket: upload_to_s3(weeks_file, bucket, prefix="") # Root upload


def create_weekly_output(winners: Dict[str, Paper], week_date: str) -> dict:
    output = {
        "week_date": week_date,
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
    print("AI Research Digest - Tournament Edition")
    print("="*60)
    
    week_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    papers_by_topic = fetch_all_papers_by_topic()
    scored_papers = score_papers_detailed(papers_by_topic)
    winners = generate_summaries_for_winners(scored_papers)
    output = create_weekly_output(winners, week_date)
    
    out_file = OUT_DIR / f"{week_date}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        
    if OUTPUT_BUCKET:
        upload_to_s3(out_file, OUTPUT_BUCKET, OUTPUT_PREFIX)
    
    # FIXED: Calling without OUTPUT_PREFIX so it uploads to root
    update_weeks_index(week_date, OUTPUT_BUCKET, OUT_DIR)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    # AWS EC2 AUTO-SHUTDOWN:
    # This ensures your server turns off immediately after finishing, saving you money!
    print("Shutting down EC2 instance to save costs...")
    os.system("sudo shutdown -h now")

if __name__ == "__main__":
    main()