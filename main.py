import arxiv

search = arxiv.Search(
    query="cat:cs.AI",
    max_results=1,
    sort_by=arxiv.SortCriterion.SubmittedDate,
)
client = arxiv.Client()

r = next(client.results(search))
print(type(r))
print([a for a in dir(r) if not a.startswith("_")])
