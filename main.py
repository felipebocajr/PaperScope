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
        "max_papers": 500
    },
    "reinforcement_learning": {
        "name": "Reinforcement Learning",
        "query": '''(ti:"reinforcement learning" OR ti:RLHF OR ti:PPO OR ti:DPO OR 
                     ti:"policy optimization" OR abs:"reward model" OR abs:"Q-learning" OR
                     abs:"actor-critic")''',
        "max_papers": 500
    },
    "llms_applications": {
        "name": "LLMs & Applications",
        "query": '''(ti:LLM OR ti:"language model" OR ti:"fine-tuning" OR ti:RAG OR 
                     ti:"prompt engineering" OR abs:"retrieval augmented" OR 
                     abs:"instruction tuning" OR abs:"in-context learning")''',
        "max_papers": 500
    },
    "computer_vision": {
        "name": "Computer Vision",
        "query": '''((ti:vision OR ti:image OR ti:detection OR ti:segmentation OR 
                      ti:"object detection" OR abs:"convolutional" OR abs:"visual recognition")
                     AND NOT (ti:language OR ti:multimodal OR ti:"vision-language" OR 
                              abs:"vision-language" OR abs:"cross-modal"))''',
        "max_papers": 500
    },
    "multimodal": {
        "name": "Multimodal",
        "query": '''(ti:multimodal OR ti:"vision-language" OR ti:CLIP OR ti:"cross-modal" OR 
                     abs:"vision and language" OR abs:"audio-visual" OR abs:"video-language")''',
        "max_papers": 500
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
ARXIV_DELAY_SEC = float(os.getenv("ARXIV_DELAY_SEC", "3"))
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))

# Directories
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
OUT_DIR = Path(os.getenv("OUT_DIR", "./out"))
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# LLM parameters
PHASE1_MAX_TOKENS = 400
PHASE1_TEMPERATURE = 0.0

PHASE2_MAX_TOKENS = 512
PHASE2_TEMPERATURE = 0.0

PHASE3_MAX_TOKENS = 900
PHASE3_TEMPERATURE = 0.5
PHASE3_TOP_P = 0.8

# Regex helpers
RE_LEADING_REASONING = re.compile(r"^\s*<reasoning>.*?</reasoning>\s*", re.DOTALL)
RE_REFS_LINE = re.compile(r"^\s*(references|bibliography|works\s+cited)\b", re.IGNORECASE)


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
# Bedrock invocation
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

def build_detailed_scoring_prompt(abstract: str, title: str) -> str:
    """Phase 2: Detailed scoring using abstract only."""
    return f"""You are evaluating a research paper for an AI/ML digest.

IMPORTANT: You are evaluating recently published research from {datetime.utcnow().strftime('%B %Y')}. Do not penalize papers for mentioning newer tools or models.

Score this paper on 3 criteria (each 0.0-10.0, use decimals):

1. innovation (weight 50%): Does it challenge assumptions or introduce genuinely new approaches?

2. impact (weight 30%): Performance improvements? Real-world application potential?

3. methodology (weight 20%): Well-explained? Reproducible? Solid approach?

Return ONLY valid JSON:
{{
  "innovation": <0.0-10.0>,
  "impact": <0.0-10.0>,
  "methodology": <0.0-10.0>,
  "reasoning": "<2-3 sentences explaining scores>"
}}

Title: {title}

Abstract:
{abstract}

JSON:""".strip()


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
        time.sleep(1)  # Polite delay between topic queries
    
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
# Phase 2: Detailed scoring (now Phase 1 in new flow)
# -------------------------

def score_papers_detailed(papers_by_topic: Dict[str, List[Paper]], top_k: int = TOP_K_PER_TOPIC, model_id: str = GPT_OSS_120B_MODEL_ID) -> Dict[str, List[Paper]]:
    """Score papers in detail (innovation/impact/methodology) and pick top K per topic."""
    print(f"\nDetailed scoring (top {top_k} per topic)...")
    print(f"Using model: {model_id}")
    
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    scored_by_topic = {}
    
    for topic_key, topic_papers in papers_by_topic.items():
        topic_name = TOPIC_QUERIES[topic_key]["name"]
        
        if not topic_papers:
            print(f"\n{topic_name}: No papers")
            scored_by_topic[topic_key] = []
            continue
        
        print(f"\n{topic_name}: Scoring {len(topic_papers)} papers...")
        
        for paper in topic_papers:
            try:
                prompt = build_detailed_scoring_prompt(paper.abstract, paper.title)
                
                response_text = invoke_bedrock(
                    bedrock,
                    model_id,
                    prompt,
                    max_tokens=PHASE2_MAX_TOKENS,
                    temperature=PHASE2_TEMPERATURE,
                )
                
                data = json.loads(response_text)
                paper.innovation = float(data.get("innovation", 0))
                paper.impact = float(data.get("impact", 0))
                paper.methodology = float(data.get("methodology", 0))
                paper.detailed_reasoning = data.get("reasoning", "")
                
                # Calculate weighted score
                paper.weighted_score = (
                    paper.innovation * 0.5 +
                    paper.impact * 0.3 +
                    paper.methodology * 0.2
                )
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  Error: {e}")
                paper.weighted_score = 0.0
        
        # Sort by weighted score and take top K
        topic_papers.sort(key=lambda p: p.weighted_score, reverse=True)
        top_papers = topic_papers[:top_k]
        
        print(f"  Top {len(top_papers)} papers:")
        for i, paper in enumerate(top_papers, 1):
            print(f"    {i}. [{paper.weighted_score:.2f}] {paper.title[:60]}")
        
        scored_by_topic[topic_key] = top_papers
    
    print("\n✓ Detailed scoring complete")
    return scored_by_topic


# -------------------------
# Phase 3: Generate summaries (now Phase 2 in new flow)
# -------------------------

def generate_summaries_for_winners(papers_by_topic: Dict[str, List[Paper]], model_id: str = GPT_OSS_120B_MODEL_ID) -> Dict[str, Paper]:
    """Generate summaries for the best paper in each topic."""
    print(f"\nGenerating summaries (1 per topic)...")
    print(f"Using model: {model_id}")
    
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
        "generated_at": datetime.utcnow().isoformat() + "Z",
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
# S3 upload helper
# -------------------------

def upload_output_file_to_s3(local_path: Path):
    """Optionally push the JSON file to S3 if OUTPUT_BUCKET is configured."""
    bucket = os.getenv("OUTPUT_BUCKET", "").strip()
    if not bucket:
        print("S3 upload disabled (OUTPUT_BUCKET not set)")
        return
    prefix = os.getenv("OUTPUT_PREFIX", "").strip()
    # build key with optional prefix
    if prefix:
        # ensure no leading/trailing slashes on prefix
        prefix = prefix.strip("/")
        key = f"{prefix}/{local_path.name}"
    else:
        key = local_path.name

    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        print(f"Uploading {local_path} to s3://{bucket}/{key} ...")
        s3.upload_file(str(local_path), bucket, key)
        print(f"✓ Uploaded {local_path.name} to s3://{bucket}/{key}")
    except Exception as e:
        print(f"Error uploading to S3: {e}")


# -------------------------
# Main
# -------------------------

def main():
    print("="*60)
    print("AI Research Digest - Topic Query Pipeline")
    print("="*60)
    
    week_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch papers by topic using targeted queries
    papers_by_topic = fetch_all_papers_by_topic()
    
    # Score papers in detail and pick top K per topic
    scored_papers = score_papers_detailed(papers_by_topic)
    
    # Generate summaries for winners
    winners = generate_summaries_for_winners(scored_papers)
    
    # Create output
    output = create_weekly_output(winners, week_date)
    
    # Save
    out_file = OUT_DIR / f"digest_{week_date}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # optionally upload to S3 if configured
    upload_output_file_to_s3(out_file)
    
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