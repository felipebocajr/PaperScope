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

AWS_REGION = os.getenv("AWS_REGION")
MODEL_ID = "openai.gpt-oss-120b-1:0"

# Optional: S3 Upload
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX")

# Topics with search queries AND descriptions for the LLM
TOPIC_QUERIES = {
    "agentic_ai": {
        "name": "Agentic AI",
        "description": "Research involving autonomous agents, tool use, multi-agent systems, and complex planning.",
        "query": '''(ti:agent OR ti:agentic OR ti:"tool use" OR ti:planning OR 
                     ti:"multi-agent" OR abs:"autonomous agent" OR abs:"tool calling" OR 
                     abs:"action space")''',
        "max_papers": 300
    },
    "reinforcement_learning": {
        "name": "Reinforcement Learning",
        "description": "Research on RL, RLHF, policy optimization, and reward modeling.",
        "query": '''(ti:"reinforcement learning" OR ti:RLHF OR ti:PPO OR ti:DPO OR 
                     ti:"policy optimization" OR abs:"reward model" OR abs:"Q-learning" OR
                     abs:"actor-critic")''',
        "max_papers": 300
    },
    "llms_applications": {
        "name": "LLMs & Applications",
        "description": "Research concerning large language models, prompt engineering, RAG, and fine-tuning.",
        "query": '''(ti:LLM OR ti:"language model" OR ti:"fine-tuning" OR ti:RAG OR 
                     ti:"prompt engineering" OR abs:"retrieval augmented" OR 
                     abs:"instruction tuning" OR abs:"in-context learning")''',
        "max_papers": 300
    },
    "computer_vision": {
        "name": "Computer Vision",
        "description": "Research focusing purely on visual recognition, image segmentation, and object detection.",
        "query": '''((ti:vision OR ti:image OR ti:detection OR ti:segmentation OR 
                      ti:"object detection" OR abs:"convolutional" OR abs:"visual recognition")
                     AND NOT (ti:language OR ti:multimodal OR ti:"vision-language"))''',
        "max_papers": 300
    },
    "multimodal": {
        "name": "Multimodal",
        "description": "Research combining multiple modalities like vision, language, audio, and cross-modal systems.",
        "query": '''(ti:multimodal OR ti:"vision-language" OR ti:CLIP OR ti:"cross-modal" OR 
                     abs:"vision and language" OR abs:"audio-visual" OR abs:"video-language")''',
        "max_papers": 300
    },
    "industrial_ml": {
        "name": "ML for Industrial & Physical Systems",
        "description": "Machine learning applied to physical systems, industrial IoT, predictive maintenance, sensor data, and edge deployment.",
        "query": '''(
                    (ti:"predictive maintenance" OR ti:"anomaly detection" OR 
                    ti:"fault detection" OR ti:"fault diagnosis" OR
                    ti:"remaining useful life" OR ti:"condition monitoring" OR
                    ti:"industrial IoT" OR ti:"IIoT" OR
                    abs:"bearing fault" OR abs:"motor current" OR 
                    abs:"vibration signal" OR abs:"SCADA" OR
                    abs:"sensor fusion" OR abs:"equipment failure" OR
                    abs:"rotating machinery" OR abs:"gearbox" OR
                    abs:"induction motor" OR abs:"prognostics")
                    OR
                    (ti:"time series" AND (abs:"industrial" OR abs:"manufacturing" OR 
                                        abs:"machinery" OR abs:"sensor"))
                    OR
                    (ti:"edge" AND (abs:"deployment" OR abs:"inference") AND 
                                (abs:"industrial" OR abs:"embedded" OR abs:"IoT"))
                )
                AND NOT (ti:weather OR ti:climate OR ti:atmospheric OR ti:aviation OR
                        ti:"partial differential" OR ti:PDE OR ti:"Navier-Stokes" OR
                        ti:"fluid dynamics" OR ti:aerodynamic OR ti:medical OR 
                        ti:clinical OR ti:genomic OR abs:"reanalysis" OR
                        abs:"numerical solver" OR abs:"finite element")''',
        "max_papers": 300
    }
}

TOPICS = {k: v["name"] for k, v in TOPIC_QUERIES.items()}
TOPIC_DISPLAY_NAMES = TOPICS

ARXIV_QUERY = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL)"
MAX_RESULTS = 1000
DAYS_LOOKBACK = 7
TOP_K_PER_TOPIC = 5

# Rate limiting
ARXIV_DELAY_SEC = 5
HTTP_TIMEOUT_SEC = 60

# Directories
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
OUT_DIR = Path(os.getenv("OUT_DIR", "./out"))
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# LLM parameters
PHASE1_MAX_TOKENS = 500
PHASE1_TEMPERATURE = 0.0

PHASE2_MAX_TOKENS = 4500
PHASE2_TEMPERATURE = 0.5 

PHASE3_MAX_TOKENS = 3250
PHASE3_TEMPERATURE = 0.3
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
    topic_fit: float = Field(ge=0.0, le=10.0, description="Score for how well the paper fits the specific topic category.")

class SummaryGlossaryOutput(BaseModel):
    summary: str = Field(description="The markdown summary of the paper.")
    glossary: Dict[str, str] = Field(description="A dictionary of complex terms (e.g. 'MARL') and their detailed explanations.")


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
    topic_fit: float = 0.0
    weighted_score: float = 0.0
    detailed_reasoning: str = ""
    
    # Phase 3: Summary & Glossary
    final_summary: str = ""
    glossary: Optional[Dict[str, str]] = None


# -------------------------
# Math Logic
# -------------------------

def calculate_final_score(innovation: float, impact: float, methodology: float, topic_fit: float) -> float:
    """Calculates weighted score and applies topic_fit penalty multiplier."""
    base_score = (innovation * 0.35) + (impact * 0.35) + (methodology * 0.30)
    # topic_fit acts as a multiplier: 10 = no penalty, 5 = 25% penalty, 0 = 50% penalty
    topic_multiplier = 0.5 + (topic_fit / 10.0) * 0.5
    return round(base_score * topic_multiplier, 2)


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

def build_comparative_batch_scoring_prompt(papers_batch: List[Paper], topic_name: str, topic_description: str) -> str:
    """PASS 1: Score a batch of papers quickly to filter out the noise."""
    papers_text = ""
    for i, paper in enumerate(papers_batch, 1):
        papers_text += f"\n{'='*60}\nPAPER {i}:\nTitle: {paper.title}\n\nAbstract:\n{paper.abstract}\n"
    
    num_papers = len(papers_batch)
    return f"""You are an expert AI/ML research evaluator. Evaluate this batch of papers.

This batch was fetched for the topic: **{topic_name}**
Topic description: {topic_description}

SCORING CRITERIA (0.0-10.0 scale):
- Innovation: Does the paper challenge assumptions or introduce genuinely novel approaches?
- Impact: Performance improvements? Real-world application potential?
- Methodology: Well-explained? Reproducible? Solid technical approach?
- Topic Fit: How well does this paper belong to '{topic_name}'? A paper about prompt injection evaluated under 'Agentic AI' should score lower than one directly about agent architectures.

OUTPUT_FORMAT & CRITICAL RULES:
1. DO NOT use double quotes (") inside the reasoning text. Use single quotes (') instead.
2. DO NOT use newlines inside the reasoning text.

Return ONLY a valid JSON array with exactly {num_papers} objects in this format:
[
  {{
    "paper_number": 1,
    "reasoning": "[Your comparative analysis for this paper, 2-3 sentences]",
    "innovation": <score 0.0-10.0>,
    "impact": <score 0.0-10.0>,
    "methodology": <score 0.0-10.0>,
    "topic_fit": <score 0.0-10.0>
  }}
]

PAPERS TO EVALUATE:
{papers_text}

JSON array:""".strip()


def build_finals_scoring_prompt(papers_batch: List[Paper], topic_name: str, topic_description: str) -> str:
    """PASS 2: The Finals. Strictly rank the top papers against each other to fix the baseline."""
    papers_text = ""
    for i, paper in enumerate(papers_batch, 1):
        papers_text += f"\n{'='*60}\nPAPER {i}:\nTitle: {paper.title}\n\nAbstract:\n{paper.abstract}\n"
    
    num_papers = len(papers_batch)
    return f"""You are an expert AI/ML research evaluator judging the "Finals" for a weekly digest.
These {num_papers} papers are the absolute best papers of the week for the topic: **{topic_name}**.

Topic description: {topic_description}

YOUR TASK:
Read all {num_papers} abstracts and rank them STRICTLY against each other. 
You must establish a global baseline: the absolute best paper in this group MUST get the highest score, and the relatively weakest in this group must get the lowest.
Force a spread in the scores. Do NOT give multiple papers the exact same scores.

SCORING CRITERIA (0.0-10.0 scale, use decimals):
- Innovation: Novelty and paradigm-shifting ideas.
- Impact: Real-world utility and performance gains.
- Methodology: Rigor and reproducibility.
- Topic Fit: Does it truly fit the {topic_name} category? Penalize papers that are excellent overall but off-topic.

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
    "methodology": <score 0.0-10.0>,
    "topic_fit": <score 0.0-10.0>
  }}
]

PAPERS TO EVALUATE:
{papers_text}

JSON array:""".strip()


def build_summary_with_glossary_prompt(full_text: str, title: str) -> str:
    """Phase 3: Generate accessible summary and extract a technical glossary."""
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
- Practical relevance: only include this if the work has genuine, near-term applicability for 
  practitioners building real systems or making technology decisions. If the paper is primarily 
  theoretical, a foundational benchmark, or an academic stepping stone, end on the results or 
  broader significance instead — do not force a "for practitioners" closing paragraph.

The order and emphasis should fit the paper - don't force a formula.
</content_requirements>

<style_guidance>
- Professional and precise, but not overly academic
- More akin to a technical blog (Distill, The Gradient) than a research abstract
- Avoid overused phrases: "game-changer," "revolutionary," "unlock," "leverage"
- Use active voice and varied sentence structure
- Use markdown formatting naturally: **bold** for genuinely critical terms or standout numbers, *italics* for emphasis.
</style_guidance>

<output_format>
You MUST return ONLY a valid JSON object. Do not wrap it in markdown code blocks if possible.
The JSON must contain two keys:
1. "summary": A 220-260 word markdown summary (3-4 paragraphs) following the style guidance above.
2. "glossary": Identify 3-6 complex acronyms or technical concepts used in your summary. Provide a clear, detailed 2-sentence explanation for each.
   CRITICAL: The keys in this dictionary MUST be exact, case-sensitive substrings of the text in your summary.

Format:
{{
  "summary": "Your markdown formatted summary here...",
  "glossary": {{
    "TERM 1": "Detailed 2-sentence explanation of term 1.",
    "TERM 2": "Detailed 2-sentence explanation of term 2."
  }}
}}
</output_format>

Title: {title}

Full paper content:
{full_text}

JSON Output:""".strip()


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

def score_papers_detailed(papers_by_topic: Dict[str, List[Paper]], top_k: int = TOP_K_PER_TOPIC, model_id: str = MODEL_ID) -> Dict[str, List[Paper]]:
    """Two-Pass Tournament Scoring: Qualifiers -> Finals"""
    print(f"\nStarting Two-Pass Tournament Scoring...")
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    scored_by_topic = {}
    
    BATCH_SIZE = 5
    QUALIFIER_TOP_K = 10
    
    for topic_key, topic_papers in papers_by_topic.items():
        topic_name = TOPIC_QUERIES[topic_key]["name"]
        topic_desc = TOPIC_QUERIES[topic_key]["description"]
        
        if not topic_papers:
            scored_by_topic[topic_key] = []
            continue
            
        print(f"\n{topic_name} [QUALIFIERS]: Evaluating {len(topic_papers)} papers...")
        
        # --- PASS 1: QUALIFIERS ---
        for batch_start in range(0, len(topic_papers), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(topic_papers))
            batch = topic_papers[batch_start:batch_end]
            
            try:
                prompt = build_comparative_batch_scoring_prompt(batch, topic_name, topic_desc)
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
                        paper.topic_fit = score_obj.topic_fit
                        paper.detailed_reasoning = score_obj.reasoning
                        
                        # Apply new logic with topic multiplier
                        paper.weighted_score = calculate_final_score(
                            paper.innovation, paper.impact, paper.methodology, paper.topic_fit
                        )
            except Exception as e:
                for paper in batch: paper.weighted_score = 0.0
        
        # Sort and pick Finalists
        topic_papers.sort(key=lambda p: p.weighted_score, reverse=True)
        finalists = topic_papers[:QUALIFIER_TOP_K]
        
        # --- PASS 2: THE FINALS ---
        print(f"  🏆 [THE FINALS] Calibrating the Top {len(finalists)} papers against each other...")
        try:
            prompt = build_finals_scoring_prompt(finalists, topic_name, topic_desc)
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
                    paper.topic_fit = score_obj.topic_fit
                    paper.detailed_reasoning = score_obj.reasoning
                    
                    # Apply new logic with topic multiplier
                    paper.weighted_score = calculate_final_score(
                        paper.innovation, paper.impact, paper.methodology, paper.topic_fit
                    )
        except Exception as e:
            print(f"  Error in Finals: {e} - Falling back to Qualifier scores.")
            
        # Final sort after calibration
        finalists.sort(key=lambda p: p.weighted_score, reverse=True)
        top_papers = finalists[:top_k]
        
        print(f"\n  Final Top {len(top_papers)} papers:")
        for i, paper in enumerate(top_papers, 1):
            print(f"    {i}. [{paper.weighted_score:.2f}] (I:{paper.innovation:.1f} M:{paper.methodology:.1f} Fit:{paper.topic_fit:.1f}) {paper.title[:50]}")
        
        scored_by_topic[topic_key] = top_papers
    
    return scored_by_topic


# -------------------------
# Phase 3: Generate summaries
# -------------------------

def generate_summaries_for_winners(papers_by_topic: Dict[str, List[Paper]], model_id: str = MODEL_ID) -> Dict[str, Paper]:
    print(f"\nGenerating summaries & glossaries (1 per topic)...")
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    winners = {}
    
    for topic_key, topic_papers in papers_by_topic.items():
        if not topic_papers: continue
        
        best_paper = max(topic_papers, key=lambda p: p.weighted_score)
        print(f"\n{TOPIC_DISPLAY_NAMES[topic_key]}:\n  Winner: {best_paper.title[:70]}")
        
        try:
            html_content = download_and_extract_html(best_paper)
            if not html_content: html_content = best_paper.abstract
            
            prompt = build_summary_with_glossary_prompt(html_content, best_paper.title)
            response_text = invoke_bedrock(bedrock, model_id, prompt, max_tokens=PHASE3_MAX_TOKENS, temperature=PHASE3_TEMPERATURE, top_p=PHASE3_TOP_P)
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            # FIX 3: Robust fallback logic. 
            if json_match:
                try:
                    parsed = SummaryGlossaryOutput.model_validate_json(json_match.group(0))
                    best_paper.final_summary = parsed.summary
                    best_paper.glossary = parsed.glossary
                except ValidationError as ve:
                    print(f"  JSON Validation failed: {ve}")
                    # If JSON parsing fails (e.g. truncated), safely fallback to paper abstract
                    best_paper.final_summary = best_paper.abstract
                    best_paper.glossary = {}
            else:
                # If no JSON was matched at all, ensure we don't return an empty string
                best_paper.final_summary = response_text if response_text.strip() else best_paper.abstract
                best_paper.glossary = {}
                
            winners[topic_key] = best_paper
            
        except Exception as e:
            print(f"  Error generating summary for {best_paper.title}: {e}")
            best_paper.final_summary = best_paper.abstract
            best_paper.glossary = {}
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


def create_weekly_output(winners: Dict[str, Paper], week_range: str) -> dict:
    output = {
        "week_date": week_range, # Mantém a chave para compatibilidade, mas agora guarda o range
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": MODEL_ID,
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
                "glossary": paper.glossary,
                "scores": {
                    "innovation": paper.innovation,
                    "impact": paper.impact,
                    "methodology": paper.methodology,
                    "topic_fit": paper.topic_fit,
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
    
    # NOVA LÓGICA DE DATAS: Gerando o range da semana
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=DAYS_LOOKBACK)
    week_range = f"{start_dt.strftime('%Y-%m-%d')}_to_{end_dt.strftime('%Y-%m-%d')}"
    
    papers_by_topic = fetch_all_papers_by_topic()
    scored_papers = score_papers_detailed(papers_by_topic)
    winners = generate_summaries_for_winners(scored_papers)
    output = create_weekly_output(winners, week_range)
    
    # Salva usando o formato de range: ex "2026-02-22_to_2026-03-01.json"
    out_file = OUT_DIR / f"{week_range}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        
    if OUTPUT_BUCKET:
        upload_to_s3(out_file, OUTPUT_BUCKET, OUTPUT_PREFIX)
    
    update_weeks_index(week_range, OUTPUT_BUCKET, OUT_DIR)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    print("Shutting down EC2 instance to save costs...")
    os.system("sudo shutdown -h now")

if __name__ == "__main__":
    main()