import streamlit as st
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="PaperScope - Paper Collection",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data


def load_weeks():
    """Load weekly reports from JSON files under data/weeks."""
    import glob
    import os

    weeks = []
    for path in sorted(glob.glob(os.path.join('data', 'weeks', '*.json'))):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                weeks.append(data)
        except FileNotFoundError:
            continue
    return weeks


# Load weekly reports
weeks = load_weeks()

if not weeks:
    st.error("❌ No weekly report files found!")
    st.info("Run `python main.py` or add reports under data/weeks.")
    st.stop()

# Title and description
st.title("📚 PaperScope - Weekly Reports")
st.markdown("A collection of weekly paper reports, grouped by week.")

# Sidebar filters (optional) - not used for weekly reports yet
# (Could add date range or score filters later)

# Display weeks and papers
for week in weeks:
    week_date = week.get('week_date', 'Unknown date')
    papers = week.get('papers', [])
    st.header(f"Week: {week_date} ({len(papers)} papers)")

    for idx, paper in enumerate(papers, 1):
        with st.container(border=True):
            # Title and optional ID
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"### {idx}. {paper.get('title','Untitled')}")
            with col2:
                arxiv_id = paper.get('arxiv_id', '') or paper.get('entry_id','').split('/abs/')[-1]
                if arxiv_id:
                    st.caption(f"arXiv:{arxiv_id}")

            # Scores or metadata if available
            scores = paper.get('scores', {})
            if scores:
                score_text = ", ".join(f"{k}: {v}" for k, v in scores.items())
                st.caption(f"Scores: {score_text}")

            # Summary
            with st.expander("📖 View Summary"):
                st.write(paper.get("summary", "No summary"))

            # Additional info (authors, categories)
            authors = paper.get("authors", [])
            if authors:
                authors_text = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_text += f", +{len(authors) - 3} more"
                st.markdown(f"**Authors:** {authors_text}")

            categories = paper.get('categories', [])
            if categories:
                st.caption(f"**Categories:** {', '.join(categories)}")

            pdf_url = paper.get("pdf_url", "")
            if pdf_url:
                st.link_button("📄 PDF", pdf_url)

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
