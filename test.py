import arxiv

client = arxiv.Client()

search = arxiv.Search(
    query="cat:cs.AI",
    max_results=1,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for r in client.results(search):
    print(r.get_short_id)
    break

