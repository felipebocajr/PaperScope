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
def load_papers():
    """Load papers from JSON file."""
    try:
        with open('data/papers.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# Load papers
papers = load_papers()

if papers is None:
    st.error("❌ Papers file not found!")
    st.info("Run `python main.py` first to collect papers from arXiv.")
    st.stop()

# Title and description
st.title("📚 PaperScope")
st.markdown("A collection of recent research papers from arXiv")

# Sidebar filters
st.sidebar.header("Filters")
selected_categories = st.sidebar.multiselect(
    "Filter by Category",
    options=sorted(set(cat for paper in papers for cat in paper.get('categories', []))),
    default=None
)

# Filter papers
if selected_categories:
    filtered_papers = [
        paper for paper in papers 
        if any(cat in selected_categories for cat in paper.get('categories', []))
    ]
else:
    filtered_papers = papers

# Summary stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Papers", len(filtered_papers))
with col2:
    authors = set(author for paper in filtered_papers for author in paper.get('authors', []))
    st.metric("Authors", len(authors))
with col3:
    categories = set(cat for paper in filtered_papers for cat in paper.get('categories', []))
    st.metric("Categories", len(categories))

st.divider()

# Display papers
st.subheader(f"Papers ({len(filtered_papers)})")

for idx, paper in enumerate(filtered_papers, 1):
    with st.container(border=True):
        # Title and ID
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### {paper['title']}")
        with col2:
            arxiv_id = paper.get("entry_id", "").split("/abs/")[-1]
            st.caption(f"arXiv:{arxiv_id}")
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"**Published:** {paper.get('published', 'N/A')}")
        with col2:
            st.caption(f"**Category:** {paper.get('primary_category', 'N/A')}")
        with col3:
            st.caption(f"**Authors:** {len(paper.get('authors', []))} authors")
        
        # Authors
        authors = paper.get("authors", [])
        if authors:
            authors_text = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_text += f", +{len(authors) - 3} more"
            st.markdown(f"**Authors:** {authors_text}")
        
        # Summary
        with st.expander("📖 View Summary"):
            st.write(paper.get("summary", "No summary"))
        
        # Categories and PDF
        col1, col2 = st.columns([3, 1])
        with col1:
            categories = ", ".join(paper.get('categories', []))
            st.caption(f"**Categories:** {categories}")
        with col2:
            pdf_url = paper.get("pdf_url", "")
            if pdf_url:
                st.link_button("📄 PDF", pdf_url)

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
