import arxiv
import json
import os
import re
from datetime import datetime

import boto3
import requests
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# Defaults (you can override per-call)
DEFAULT_INFERENCE_PROFILE = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_SCORING_MODEL_ID = "openai.gpt-oss-120b-1:0"
SONNET_INFERENCE_PROFILE = "arn:aws:bedrock:us-east-1:836823680505:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Regex helpers
RE_LEADING_REASONING = re.compile(r"^\s*<reasoning>.*?</reasoning>\s*", re.DOTALL)


# -------------------------
# Helpers
# -------------------------

def download_pdf(url: str, filename: str) -> str:
    """Download a PDF from a URL and save locally."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename


def extract_text_from_pdf(filename: str) -> str:
    """Extract text from PDF using pypdf."""
    reader = PdfReader(filename)
    text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        text.append(t)
    return "\n".join(text)


def find_references_index(text: str) -> int:
    """Try to cut off references/bibliography to reduce noise."""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        s = line.strip().lower()
        if s in ("references", "bibliography", "works cited"):
            return idx
    return -1


def maybe_strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return s


# -------------------------
# Bedrock invocation
# -------------------------

def invoke_bedrock(
    bedrock_client,
    model_id: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    """Call Bedrock and return the text response.

    Supports:
      - Anthropic Claude models (Anthropic Messages schema)
      - OpenAI GPT-OSS models on Bedrock (OpenAI chat-completions schema)

    Notes:
      - Bedrock InvokeModel may prefix reasoning wrapped in <reasoning> tags.
      - This function strips that prefix for OpenAI models.
    """

    # ---- OpenAI GPT-OSS models (OpenAI chat-completions style body) ----
    if model_id.startswith("openai."):
        native_request = {
            "messages": [
                {"role": "system", "content": "Return only what the user asks for."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens,
            "temperature": 0.0 if temperature is None else float(temperature),
            "top_p": 1.0 if top_p is None else float(top_p),
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

        # Strip leading reasoning if present
        actual_text = RE_LEADING_REASONING.sub("", actual_text).strip()
        actual_text = maybe_strip_code_fence(actual_text)
        return actual_text

    # ---- Anthropic models (your existing schema) ----
    anthropic_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    if temperature is not None:
        anthropic_body["temperature"] = float(temperature)
    if top_p is not None:
        anthropic_body["top_p"] = float(top_p)

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
    actual_text = maybe_strip_code_fence(actual_text)
    return actual_text


# -------------------------
# Prompt builders
# -------------------------

def build_scoring_prompt(title: str, abstract: str) -> str:
    return f"""
You are evaluating an arXiv paper for inclusion in a weekly AI/ML digest.

Given the title and abstract, return ONLY valid JSON matching this exact format:
{{"score": <float between 0.0 and 10.0, using one decimal place>, "reasoning": "<1-2 sentences>"}}

Example output:
{{"score": 8.7, "reasoning": "This paper introduces a highly novel architecture that significantly improves training efficiency."}}

Scoring guidelines:
- Use decimal values (e.g., 7.4, 8.9) to provide precise, nuanced rankings. Do not limit your scores to whole numbers.
- Prioritize papers whose primary focus is developing or researching AI/ML methods, architectures, training techniques, evaluation approaches, or theoretical foundations.
- Papers that apply existing ML methods to domain-specific problems (without methodological innovation) should receive moderate scores.
- If the abstract is vague or lacks clear AI/ML research contribution, score lower.

Title: {title}
Abstract: {abstract}

Return ONLY JSON:
""".strip()


def build_summary_prompt(paper_text: str, title: str) -> str:
    return f"""
You are an insightful human curator writing a weekly AI/ML research digest for practitioners (data scientists, ML engineers, and technical founders).

<instructions>
Your task is to write a short, engaging mini-article summarizing the provided research paper.

Style & Tone:
- Write in smoothly flowing prose paragraphs. Your response must be standard text without markdown headers, numbered lists, or bullet points.
- Use active voice and vary your sentence lengths to keep the reading dynamic.
- If you mention a technical term, briefly clarify it in plain language.
- Write directly and affirmatively. Avoid clichés like "Imagine a...", "In the rapidly evolving landscape...", or "This paper proposes...".

Structure & Flow:
- Hook the reader immediately with a strong opening sentence that highlights why this specific work is interesting or what core problem it solves.
- Explain what the authors actually did differently from typical approaches. 
- Weave in at least one concrete detail (e.g., a specific scale, dataset name, or performance metric) to ground the summary in reality.
- Conclude by explaining exactly why this matters in practice for real-world engineering workflows.

Output Format:
- Keep the text concise, ideally between 160 and 220 words formatted across 2 to 4 natural paragraphs.
- You must output your final article strictly inside <article> tags. Do not output any conversational filler.
</instructions>

<paper>
Title: {title}

{paper_text}
</paper>
""".strip()


def build_detailed_scoring_prompt(paper_text: str, title: str) -> str:
    """Prompt for Phase 2: detailed scoring with 3 criteria based on full PDF."""
    return f"""
You are evaluating a research paper in detail for an AI/ML digest.

IMPORTANT: You are evaluating recently published research from {datetime.utcnow().strftime('%B %Y')}. Your training data may be outdated. Do not penalize papers for mentioning tools, models, or techniques that are newer than your knowledge cutoff. Focus on the research contribution itself.

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


def score_top_papers_with_pdf(papers, model_id: str = DEFAULT_INFERENCE_PROFILE):
    """Phase 2: Score top papers with full PDF content using 3 criteria."""
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

    detailed_scores = []
    for i, paper in enumerate(papers, 1):
        print(f"\nPhase 2: Processing paper {i}/{len(papers)}")
        print(f"Title: {paper['title'][:80]}")
        
        try:
            # Extract PDF text
            paper_text = fetch_and_extract_paper_text(paper)
            
            # Build detailed scoring prompt
            prompt = build_detailed_scoring_prompt(paper_text, paper["title"])
            
            # Call model
            response_text = invoke_bedrock(
                bedrock_runtime,
                model_id,
                prompt,
                max_tokens=512,
                temperature=0.0,
            )
            
            # Parse response
            data = json.loads(response_text)
            innovation = float(data.get("innovation", 0))
            impact = float(data.get("impact", 0))
            methodology = float(data.get("methodology", 0))
            reasoning = str(data.get("reasoning", "")).strip()
            
            # Calculate weighted score: 50% innovation, 30% impact, 20% methodology
            weighted_score = (innovation * 0.5) + (impact * 0.3) + (methodology * 0.2)
            
            print(f"Scores - Innovation: {innovation:.1f}, Impact: {impact:.1f}, Methodology: {methodology:.1f}")
            print(f"Weighted score: {weighted_score:.2f}")
            
            detailed_scores.append({
                "entry_id": paper["entry_id"],
                "innovation": innovation,
                "impact": impact,
                "methodology": methodology,
                "weighted_score": round(weighted_score, 2),
                "reasoning": reasoning,
            })
            
        except Exception as e:
            print(f"Error processing paper: {e}")
            detailed_scores.append({
                "entry_id": paper["entry_id"],
                "innovation": 0,
                "impact": 0,
                "methodology": 0,
                "weighted_score": 0,
                "reasoning": f"Error: {str(e)}",
            })
    
    return detailed_scores


# -------------------------
# Pipeline steps
# -------------------------

def fetch_papers(query: str, max_results: int = 100):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    for result in search.results():
        papers.append(
            {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "published": result.published.strftime("%Y-%m-%d"),
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "entry_id": result.entry_id,
            }
        )
    return papers


def score_papers_with_bedrock(papers, model_id: str = DEFAULT_SCORING_MODEL_ID):
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

    scores = []
    for paper in papers:
        prompt = build_scoring_prompt(paper["title"], paper["summary"])
        response_text = invoke_bedrock(
            bedrock_runtime,
            model_id,
            prompt,
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
        )

        try:
            data = json.loads(response_text)
            score = float(data.get("score", 0))
            reasoning = str(data.get("reasoning", "")).strip()
        except Exception:
            score = 0.0
            reasoning = "Failed to parse model JSON response."

        scores.append({"entry_id": paper["entry_id"], "score": score, "reasoning": reasoning})

    return scores


def fetch_and_extract_paper_text(paper: dict, out_dir: str = "./papers") -> str:
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, paper["entry_id"].split("/")[-1] + ".pdf")
    download_pdf(paper["pdf_url"], pdf_path)
    text = extract_text_from_pdf(pdf_path)

    # Attempt to cut off references
    ref_idx = find_references_index(text)
    if ref_idx != -1:
        lines = text.splitlines()
        text = "\n".join(lines[:ref_idx])

    return text


def generate_summaries_for_papers(papers, model_id: str = DEFAULT_INFERENCE_PROFILE):
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))

    for paper in papers:
        paper_text = fetch_and_extract_paper_text(paper)
        prompt = build_summary_prompt(paper_text, paper["title"])

        summary_text = invoke_bedrock(
            bedrock_runtime,
            model_id,
            prompt,
            max_tokens=900,
        )

        paper["summary"] = summary_text

    return papers


# -------------------------
# Entry point
# -------------------------

def main():
    query = os.getenv("ARXIV_QUERY", "cat:cs.AI")
    max_results = int(os.getenv("ARXIV_MAX_RESULTS", "50"))

    # Phase 1: Fetch and score papers with GPT-OSS 120B
    print("="*60)
    print("PHASE 1: Fetching and scoring papers with GPT-OSS 120B")
    print("="*60)
    papers = fetch_papers(query=query, max_results=max_results)
    print(f"Fetched {len(papers)} papers")

    scores = score_papers_with_bedrock(papers, model_id=DEFAULT_SCORING_MODEL_ID)

    # Attach scores
    score_map = {s["entry_id"]: s for s in scores}
    for p in papers:
        p["score"] = score_map.get(p["entry_id"], {}).get("score", 0)
        p["score_reasoning"] = score_map.get(p["entry_id"], {}).get("reasoning", "")

    # Sort by score descending and take top 10
    papers.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_10_papers = papers[:10]

    print(f"\nTop 10 papers from Phase 1:")
    for i, p in enumerate(top_10_papers, 1):
        print(f"{i}. [{p['score']:.1f}] {p['title'][:80]}")

    # Phase 2: Detailed scoring of top 10 with Haiku 4.5 (PDF analysis)
    print("\n" + "="*60)
    print("PHASE 2: Detailed scoring of top 10 papers with Haiku 4.5")
    print("="*60)
    detailed_scores = score_top_papers_with_pdf(top_10_papers, model_id=DEFAULT_INFERENCE_PROFILE)

    # Attach detailed scores to papers
    detailed_score_map = {s["entry_id"]: s for s in detailed_scores}
    for p in top_10_papers:
        details = detailed_score_map.get(p["entry_id"], {})
        p["innovation"] = details.get("innovation", 0)
        p["impact"] = details.get("impact", 0)
        p["methodology"] = details.get("methodology", 0)
        p["weighted_score"] = details.get("weighted_score", 0)
        p["detailed_reasoning"] = details.get("reasoning", "")

    # Sort by weighted score and take top 3
    top_10_papers.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
    top_3_papers = top_10_papers[:3]

    print(f"\n" + "="*60)
    print("TOP 3 PAPERS SELECTED (after Phase 2)")
    print("="*60)
    for i, p in enumerate(top_3_papers, 1):
        print(f"\n{i}. [Weighted: {p['weighted_score']:.2f}] {p['title'][:80]}")
        print(f"   Innovation: {p['innovation']:.1f} | Impact: {p['impact']:.1f} | Methodology: {p['methodology']:.1f}")

    # Phase 3a: Generate summaries with GPT-OSS
    print("\n" + "="*60)
    print("PHASE 3a: Generating summaries with GPT-OSS 120B")
    print("="*60)
    papers_gpt = [p.copy() for p in top_3_papers]
    papers_gpt = generate_summaries_for_papers(papers_gpt, model_id=DEFAULT_SCORING_MODEL_ID)

    out_gpt = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": "GPT-OSS 120B",
        "query": query,
        "papers": papers_gpt,
    }

    out_path_gpt = "weekly_digest_gpt.json"
    with open(out_path_gpt, "w", encoding="utf-8") as f:
        json.dump(out_gpt, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path_gpt}")

    # Phase 3b: Generate summaries with Sonnet 4.5
    print("\n" + "="*60)
    print("PHASE 3b: Generating summaries with Sonnet 4.5")
    print("="*60)
    papers_sonnet = [p.copy() for p in top_3_papers]
    papers_sonnet = generate_summaries_for_papers(papers_sonnet, model_id=SONNET_INFERENCE_PROFILE)

    out_sonnet = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": "Claude Sonnet 4.5",
        "query": query,
        "papers": papers_sonnet,
    }

    out_path_sonnet = "weekly_digest_sonnet.json"
    with open(out_path_sonnet, "w", encoding="utf-8") as f:
        json.dump(out_sonnet, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path_sonnet}")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Compare the two outputs:")
    print(f"  - GPT-OSS: {out_path_gpt}")
    print(f"  - Sonnet:  {out_path_sonnet}")



if __name__ == "__main__":
    main()