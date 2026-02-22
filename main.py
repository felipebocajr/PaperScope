import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import arxiv
import boto3
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# -------------------------
# Configuration
# -------------------------

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Use GPT-OSS 120B for all phases (per your updated design)
GPT_OSS_120B_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")

# Pipeline size
ARXIV_QUERY = os.getenv("ARXIV_QUERY", "cat:cs.AI")
MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "1000"))
DAYS_LOOKBACK = int(os.getenv("ARXIV_DAYS_LOOKBACK", "7"))
TOP_K_PHASE1 = int(os.getenv("TOP_K_PHASE1", "10"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "3"))

# arXiv polite rate limiting: >= 3 seconds between API calls is recommended
ARXIV_DELAY_SEC = float(os.getenv("ARXIV_DELAY_SEC", "3"))

# Storage / caching
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
PAPERS_DIR = os.path.join(CACHE_DIR, "papers")
EXTRACT_DIR = os.path.join(CACHE_DIR, "extracted")
OUT_DIR = os.getenv("OUT_DIR", "./out")

# S3 output (optional)
OUTPUT_BUCKET = os.getenv("OUTPUT_BUCKET", "")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "weeks")

# HTML extraction
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "60"))
PREFER_HTML = os.getenv("PREFER_HTML", "1") == "1"

# Token controls
PHASE1_MAX_TOKENS = int(os.getenv("PHASE1_MAX_TOKENS", "220"))
PHASE2_MAX_TOKENS = int(os.getenv("PHASE2_MAX_TOKENS", "420"))
PHASE3_MAX_TOKENS = int(os.getenv("PHASE3_MAX_TOKENS", "900"))

PHASE1_TEMPERATURE = float(os.getenv("PHASE1_TEMPERATURE", "0.0"))
PHASE2_TEMPERATURE = float(os.getenv("PHASE2_TEMPERATURE", "0.0"))
PHASE3_TEMPERATURE = float(os.getenv("PHASE3_TEMPERATURE", "0.7"))

PHASE1_TOP_P = float(os.getenv("PHASE1_TOP_P", "1.0"))
PHASE2_TOP_P = float(os.getenv("PHASE2_TOP_P", "1.0"))
PHASE3_TOP_P = float(os.getenv("PHASE3_TOP_P", "0.9"))


# -------------------------
# Regex helpers
# -------------------------

RE_LEADING_REASONING = re.compile(r"^\s*<reasoning>.*?</reasoning>\s*", re.DOTALL)
RE_REFS_LINE = re.compile(r"^\s*(references|bibliography|works\s+cited)\b", re.IGNORECASE)
RE_WS = re.compile(r"[ \t]+")


# -------------------------
# Data structures
# -------------------------

@dataclass
class Paper:
    title: str
    authors: List[str]
    published: str  # YYYY-MM-DD
    abstract: str
    pdf_url: str
    entry_id: str
    arxiv_id: str

    # Scores
    phase1_score: float = 0.0
    phase1_reasoning: str = ""

    innovation: float = 0.0
    impact: float = 0.0
    methodology: float = 0.0
    weighted_score: float = 0.0
    detailed_reasoning: str = ""

    # Output summary
    final_summary: str = ""


# -------------------------
# Small utilities
# -------------------------

def ensure_dirs() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(PAPERS_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def week_date_iso(now_utc: datetime) -> str:
    """Return an ISO date label for the weekly output.

    Default: upcoming Sunday (UTC) as the 'week_date', matching your example style.
    """
    # Monday=0 ... Sunday=6
    days_until_sun = (6 - now_utc.weekday()) % 7
    label = (now_utc.date() + timedelta(days=days_until_sun)).isoformat()
    return label


def safe_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def maybe_strip_code_fence(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
    return s


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def arxiv_id_from_entry_id(entry_id: str) -> str:
    """Best-effort parse of arXiv id from entry_id URL."""
    # entry_id is like https://arxiv.org/abs/2402.12345v1
    tail = entry_id.rstrip("/").split("/")[-1]
    # keep versioned tail; we'll use v1 for html
    # but for arxiv_id we store without version
    if "v" in tail:
        base = tail.split("v")[0]
        return base
    return tail


def paper_cache_key(arxiv_id: str) -> str:
    # filesystem-safe
    return arxiv_id.replace("/", "_")


# -------------------------
# arXiv fetch
# -------------------------

def fetch_papers(query: str, max_results: int, days_lookback: int) -> List[Paper]:
    cutoff = (utc_now() - timedelta(days=days_lookback)).date()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client(
        page_size=100,
        delay_seconds=ARXIV_DELAY_SEC,  # uses the library's built-in delay
        num_retries=3,
    )

    papers: List[Paper] = []
    for idx, result in enumerate(client.results(search), 1):
        # log progress every 25 papers (roughly)
        if idx % 25 == 0:
            print(f"fetch_papers: processed {idx} results...")

        published_date = result.published.date()
        if published_date < cutoff:
            break

        entry_id = result.entry_id
        arxiv_id = arxiv_id_from_entry_id(entry_id)

        papers.append(
            Paper(
                title=result.title.strip(),
                authors=[a.name for a in result.authors],
                published=result.published.strftime("%Y-%m-%d"),
                abstract=(result.summary or "").strip(),
                pdf_url=result.pdf_url,
                entry_id=entry_id,
                arxiv_id=arxiv_id,
            )
        )

    return papers


# -------------------------
# Extraction (HTML-first, PDF fallback)
# -------------------------

def download_pdf(url: str, filename: str) -> str:
    r = requests.get(url, timeout=HTTP_TIMEOUT_SEC)
    r.raise_for_status()
    with open(filename, "wb") as f:
        f.write(r.content)
    return filename


def extract_text_from_pdf(filename: str) -> str:
    reader = PdfReader(filename)
    text_parts: List[str] = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def clean_text_block(text: str) -> str:
    """Normalize whitespace and remove obvious PDF/HTML artifacts."""
    if not text:
        return ""

    # De-hyphenate line breaks: "inter-\nnational" -> "international"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize newlines and spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = RE_WS.sub(" ", line).strip()
        if not line:
            continue
        lines.append(line)
    return "\n".join(lines)


def stop_at_references(text: str) -> str:
    lines = text.splitlines()
    out: List[str] = []
    for line in lines:
        if RE_REFS_LINE.match(line):
            break
        out.append(line)
    return "\n".join(out)


def fetch_arxiv_html(arxiv_id: str) -> str:
    # HTML pages are often versioned. Use v1 by default.
    url = f"https://arxiv.org/html/{arxiv_id}v1"
    r = requests.get(url, timeout=HTTP_TIMEOUT_SEC)
    if r.status_code == 404:
        raise FileNotFoundError(f"No HTML for {arxiv_id}")
    r.raise_for_status()
    return r.text


def html_to_text_arxiv(html: str) -> str:
    """Convert arXiv HTML to linear text using BeautifulSoup.

    This is a lightweight replacement for synthetic-data-kit.
    You can swap this implementation later.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # arXiv HTML typically has an <article> or a div with ltx_document
    main = soup.find("article")
    if main is None:
        main = soup.find("div", class_=re.compile(r"ltx_document"))
    if main is None:
        main = soup.body or soup

    # Drop nav/sidebars if any
    for tag in main.find_all(["nav", "header", "footer", "aside"]):
        tag.decompose()

    text = main.get_text("\n")
    text = clean_text_block(text)
    text = stop_at_references(text)
    return text


def fetch_and_extract_paper_content(paper: Paper) -> Dict[str, str]:
    """Extract full content for a paper.

    Prefers arXiv HTML when available (cleaner), with PDF fallback.
    Uses on-disk caching.

    Returns:
      {"source": "html"|"pdf", "text": "..."}
    """
    ensure_dirs()

    key = paper_cache_key(paper.arxiv_id)
    cache_path = os.path.join(EXTRACT_DIR, f"{key}.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 1) Try HTML
    if PREFER_HTML:
        try:
            html = fetch_arxiv_html(paper.arxiv_id)
            text = html_to_text_arxiv(html)
            payload = {"source": "html", "text": text}
            safe_write_json(cache_path, payload)
            return payload
        except Exception:
            # fall back to PDF
            pass

    # 2) PDF fallback
    pdf_path = os.path.join(PAPERS_DIR, f"{key}.pdf")
    if not os.path.exists(pdf_path):
        download_pdf(paper.pdf_url, pdf_path)

    pdf_text = extract_text_from_pdf(pdf_path)
    pdf_text = clean_text_block(pdf_text)
    pdf_text = stop_at_references(pdf_text)

    payload = {"source": "pdf", "text": pdf_text}
    safe_write_json(cache_path, payload)
    return payload


# -------------------------
# Bedrock invocation (OpenAI + Anthropic support)
# -------------------------

def invoke_bedrock(
    bedrock_client,
    model_id: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Call Bedrock and return the text response.

    Supports:
      - OpenAI GPT-OSS models on Bedrock (OpenAI chat-completions schema)
      - Anthropic Claude models (Anthropic Messages schema)
    """

    # ---- OpenAI GPT-OSS models (OpenAI chat-completions style body) ----
    if model_id.startswith("openai."):
        native_request = {
            "messages": [
                {"role": "system", "content": "Follow instructions exactly."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
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

        obj = json.loads(payload)
        text = (
            obj.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        ).strip()

        text = RE_LEADING_REASONING.sub("", text).strip()
        return maybe_strip_code_fence(text)

    # ---- Anthropic fallback ----
    anthropic_body: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "messages": [{"role": "user", "content": prompt}],
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

    obj = json.loads(payload)
    text = (obj.get("content", [{}])[0].get("text", "") or "").strip()
    return maybe_strip_code_fence(text)


# -------------------------
# Prompt builders
# -------------------------

def build_scoring_prompt(title: str, abstract: str) -> str:
    return f"""
You are selecting papers for a weekly AI/ML research digest.

Return ONLY valid JSON:
{{"score": <0.0-10.0 with exactly ONE decimal>, "reasoning": "<1-2 sentences>"}}

Rules:
- Score must be a decimal with one digit (e.g., 7.4, 8.9). Do NOT return integers.
- Use the full scale and avoid ties; separate close papers by 0.1–0.3.
- Prioritize broadly useful AI/ML methods (architecture, training, evaluation, theory).
- Lower scores for mostly application papers (e.g., robotics) unless they introduce reusable ML ideas.

Title: {title}
Abstract: {abstract}

JSON:
""".strip()


def build_detailed_scoring_prompt(full_text: str, title: str) -> str:
    month_year = utc_now().strftime("%B %Y")
    return f"""
You are evaluating a recent research paper in detail for an AI/ML digest.

IMPORTANT: The paper is from the last {DAYS_LOOKBACK} days (as of {month_year}). Your training data may be outdated.
Do not penalize the paper for mentioning tools/models newer than your cutoff; evaluate the research contribution itself.

Score on three criteria (each 0.0-10.0, decimals allowed):
1) innovation (50%): genuinely new idea, challenges assumptions, non-trivial novelty.
2) impact (30%): likely usefulness, meaningful improvement, realistic downstream potential.
3) methodology (20%): clarity, soundness, reproducibility, evidence quality.

Return ONLY valid JSON with exactly these keys:
{{"innovation": <0.0-10.0>, "impact": <0.0-10.0>, "methodology": <0.0-10.0>, "reasoning": "<2-3 sentences>"}}

Title: {title}

Paper content:
{full_text}

JSON:
""".strip()


def build_summary_prompt(full_text: str, title: str) -> str:
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
# Phase 1: generic scoring (title + abstract)
# -------------------------

def phase1_score(papers: List[Paper], bedrock_client, model_id: str) -> List[Paper]:
    scored: List[Paper] = []
    for i, p in enumerate(papers, 1):
        prompt = build_scoring_prompt(p.title, p.abstract)
        resp = invoke_bedrock(
            bedrock_client,
            model_id,
            prompt,
            max_tokens=PHASE1_MAX_TOKENS,
            temperature=PHASE1_TEMPERATURE,
            top_p=PHASE1_TOP_P,
        )

        try:
            data = json.loads(resp)
            p.phase1_score = clamp(float(data.get("score", 0.0)), 0.0, 10.0)
            p.phase1_reasoning = str(data.get("reasoning", "")).strip()
        except Exception:
            p.phase1_score = 0.0
            p.phase1_reasoning = "Failed to parse JSON from model."

        scored.append(p)

        if i % 50 == 0:
            print(f"Phase 1 scored {i}/{len(papers)}")

    scored.sort(key=lambda x: x.phase1_score, reverse=True)
    return scored


# -------------------------
# Phase 2: detailed scoring (full text via HTML/PDF extraction)
# -------------------------

def phase2_detailed_score(papers: List[Paper], bedrock_client, model_id: str) -> List[Paper]:
    out: List[Paper] = []
    for i, p in enumerate(papers, 1):
        print(f"\nPhase 2: {i}/{len(papers)} — {p.title[:80]}")
        content = fetch_and_extract_paper_content(p)
        full_text = content.get("text", "")

        # Safety cap: avoid insane context in case HTML extraction explodes
        # (You can tune this based on observed token usage.)
        if len(full_text) > 120_000:
            full_text = full_text[:120_000]

        prompt = build_detailed_scoring_prompt(full_text, p.title)
        resp = invoke_bedrock(
            bedrock_client,
            model_id,
            prompt,
            max_tokens=PHASE2_MAX_TOKENS,
            temperature=PHASE2_TEMPERATURE,
            top_p=PHASE2_TOP_P,
        )

        try:
            data = json.loads(resp)
            p.innovation = clamp(float(data.get("innovation", 0.0)), 0.0, 10.0)
            p.impact = clamp(float(data.get("impact", 0.0)), 0.0, 10.0)
            p.methodology = clamp(float(data.get("methodology", 0.0)), 0.0, 10.0)
            p.detailed_reasoning = str(data.get("reasoning", "")).strip()
        except Exception as e:
            p.innovation = p.impact = p.methodology = 0.0
            p.detailed_reasoning = f"Failed to parse JSON from model: {e}"

        p.weighted_score = round((p.innovation * 0.5) + (p.impact * 0.3) + (p.methodology * 0.2), 2)

        print(
            f"Scores — innovation {p.innovation:.1f}, impact {p.impact:.1f}, methodology {p.methodology:.1f} → weighted {p.weighted_score:.2f}"
        )

        out.append(p)

    out.sort(key=lambda x: x.weighted_score, reverse=True)
    return out


# -------------------------
# Phase 3: summary generation (full text via HTML/PDF extraction)
# -------------------------

def phase3_generate_summaries(papers: List[Paper], bedrock_client, model_id: str) -> List[Paper]:
    out: List[Paper] = []
    for i, p in enumerate(papers, 1):
        print(f"\nPhase 3: {i}/{len(papers)} — summarizing {p.title[:80]}")
        content = fetch_and_extract_paper_content(p)
        full_text = content.get("text", "")

        if len(full_text) > 120_000:
            full_text = full_text[:120_000]

        prompt = build_summary_prompt(full_text, p.title)
        resp = invoke_bedrock(
            bedrock_client,
            model_id,
            prompt,
            max_tokens=PHASE3_MAX_TOKENS,
            temperature=PHASE3_TEMPERATURE,
            top_p=PHASE3_TOP_P,
        )

        p.final_summary = resp.strip()
        out.append(p)

    return out


# -------------------------
# Output + S3 upload
# -------------------------

def save_phase1_scores(papers: List[Paper], week_date: str) -> str:
    """Save Phase 1 scoring results to JSON for inspection."""
    payload = {
        "week_date": week_date,
        "generated_at": utc_now().isoformat().replace("+00:00", "Z"),
        "query": ARXIV_QUERY,
        "total_papers": len(papers),
        "papers": [
            {
                "title": p.title,
                "arxiv_id": p.arxiv_id,
                "arxiv_url": p.entry_id,
                "published": p.published,
                "phase1_score": p.phase1_score,
                "phase1_reasoning": p.phase1_reasoning,
            }
            for p in papers
        ],
    }
    out_path = os.path.join(OUT_DIR, f"phase1_scores_{week_date}.json")
    safe_write_json(out_path, payload)
    return out_path


def build_weekly_payload(week_date: str, model_id: str, papers: List[Paper]) -> Dict[str, Any]:
    return {
        "week_date": week_date,
        "model": model_id,
        "generated_at": utc_now().isoformat().replace("+00:00", "Z"),
        "query": ARXIV_QUERY,
        "papers": [
            {
                "title": p.title,
                "summary": p.final_summary,
                "arxiv_url": p.entry_id,
                "arxiv_id": p.arxiv_id,
                "published": p.published,
                "scores": {
                    "phase1": p.phase1_score,
                    "innovation": p.innovation,
                    "impact": p.impact,
                    "methodology": p.methodology,
                    "weighted": p.weighted_score,
                },
            }
            for p in papers
        ],
    }


def maybe_upload_to_s3(local_path: str, bucket: str, key: str) -> None:
    if not bucket:
        return
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file(local_path, bucket, key)
    print(f"Uploaded to s3://{bucket}/{key}")


# -------------------------
# Main
# -------------------------

def main() -> None:
    ensure_dirs()

    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    print("=" * 70)
    print("AI Research Digest Pipeline (HTML-first, PDF fallback)")
    print("=" * 70)

    # Fetch and filter
    print(f"Fetching papers from arXiv: query={ARXIV_QUERY}, lookback={DAYS_LOOKBACK}d, max={MAX_RESULTS}")
    papers = fetch_papers(ARXIV_QUERY, MAX_RESULTS, DAYS_LOOKBACK)
    print(f"Fetched {len(papers)} papers in lookback window")

    # Phase 1
    print("\n" + "=" * 70)
    print("PHASE 1: Generic scoring (title + abstract)")
    print("=" * 70)
    papers_scored = phase1_score(papers, bedrock, GPT_OSS_120B_MODEL_ID)
    top10 = papers_scored[:TOP_K_PHASE1]

    print(f"\nTop {TOP_K_PHASE1} papers from Phase 1:")
    for i, p in enumerate(top10, 1):
        print(f"{i:02d}. [{p.phase1_score:.1f}] {p.title[:90]}")

    # Save all Phase 1 scores for inspection
    now = utc_now()
    week_date = week_date_iso(now)
    phase1_output = save_phase1_scores(papers_scored, week_date)
    print(f"\nSaved Phase 1 scores: {phase1_output}")


    # Phase 2
    print("\n" + "=" * 70)
    print(f"PHASE 2: Detailed scoring of top {TOP_K_PHASE1} (HTML-first full text)")
    print("=" * 70)
    detailed = phase2_detailed_score(top10, bedrock, GPT_OSS_120B_MODEL_ID)
    top3 = detailed[:TOP_K_FINAL]

    print("\n" + "=" * 70)
    print(f"TOP {TOP_K_FINAL} SELECTED")
    print("=" * 70)
    for i, p in enumerate(top3, 1):
        print(f"\n{i}. [Weighted {p.weighted_score:.2f}] {p.title}")
        print(f"   innovation {p.innovation:.1f} | impact {p.impact:.1f} | methodology {p.methodology:.1f}")

    # save raw html for the final three papers for manual validation
    try:
        html_path = os.path.join(OUT_DIR, "top3_html.html")
        with open(html_path, "w", encoding="utf-8") as f:
            for p in top3:
                f.write(f"<!-- {p.arxiv_id} -->\n")
                try:
                    raw_html = fetch_arxiv_html(p.arxiv_id)
                except Exception as e:
                    raw_html = f"ERROR fetching HTML: {e}"
                f.write(raw_html)
                f.write("\n\n")
        print(f"Saved HTML for top3 papers: {html_path}")
    except Exception as e:
        print(f"Failed to write top3 HTML file: {e}")

    # Phase 3
    print("\n" + "=" * 70)
    print("PHASE 3: Summary generation")
    print("=" * 70)
    summarized = phase3_generate_summaries(top3, bedrock, GPT_OSS_120B_MODEL_ID)

    # Output
    now = utc_now()
    week_date = week_date_iso(now)
    payload = build_weekly_payload(week_date, GPT_OSS_120B_MODEL_ID, summarized)

    local_week_dir = os.path.join(OUT_DIR, OUTPUT_PREFIX)
    os.makedirs(local_week_dir, exist_ok=True)
    out_path = os.path.join(local_week_dir, f"{week_date}.json")
    safe_write_json(out_path, payload)
    print(f"\nSaved weekly JSON: {out_path}")


    # Optional S3 upload
    if OUTPUT_BUCKET:
        s3_key = f"{OUTPUT_PREFIX}/{week_date}.json"
        maybe_upload_to_s3(out_path, OUTPUT_BUCKET, s3_key)

    print("\nDone.")


if __name__ == "__main__":
    main()
