# TN Government Schemes — Knowledge Graph RAG Chatbot

**Gen AI Architect Program · Week 3 Day 1 · Knowledge Graph assignment**

A chatbot that answers questions about Tamil Nadu government schemes
([source page](https://www.tn.gov.in/scheme_list.php?dep_id=Mg==)) using a
**knowledge graph + vector hybrid retrieval (GraphRAG)** instead of plain vector RAG.

## Architecture

```
 Saved scheme pages (HTML/PDF/TXT)
            │  ingest.py   – strip nav/footer, keep tables as "Label : Value"
            ▼
       pages.json
            │  build_graph.py – LLM structured extraction (Pydantic schema)
            ▼
       schemes.json ──► NetworkX Knowledge Graph ──► graph.json / graph.cypher (Neo4j)
            └─────────► FAISS index (scheme profiles + raw text chunks)
                                   │
 User question ──► retriever.py ───┤ 1. vector search → seed schemes
                                   │ 2. entity linking on graph nodes (fuzzy)
                                   │ 3. graph expansion → triples
                                   ▼
                       LLM answers from triples + source text
                                   ▼
                       app.py (Streamlit chat + graph viewer)
```

### Graph schema

| Node | Relation from `Scheme` |
|---|---|
| Department | `ADMINISTERED_BY` |
| BeneficiaryGroup | `TARGETS` |
| Eligibility | `HAS_ELIGIBILITY` |
| Benefit | `PROVIDES` |
| Document | `REQUIRES` |
| ApplicationChannel | `APPLY_VIA` |
| Sector | `COVERS` |
| Funding | `FUNDED_BY` |

Shared nodes (e.g. *small and marginal farmers*, *aadhaar card*) connect different
schemes, so the bot can answer cross-scheme questions like *"Which schemes are for
women farmers?"* or *"Which schemes need chitta and adangal?"* — questions where
plain vector RAG only returns whichever chunk happens to be most similar.

## Why GraphRAG here (for the write-up)

- **Multi-hop / aggregate questions** — "all schemes that need X" is a graph traversal, not a similarity search.
- **Precision** — amounts, eligibility and documents are stored as discrete facts, so the LLM gets clean triples instead of noisy page text.
- **Explainability** — every answer shows the triples and subgraph it used.
- **Hybrid** — vector search still handles fuzzy, free-form questions; the graph adds structure.

## Setup

```bash
python -m venv venv && venv\Scripts\activate      # Windows  (source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
copy .env.example .env                            # add your API key
```

### 1. Get the data
tn.gov.in disallows automated scraping, so save the pages from your browser:
1. Open the scheme list page → **Ctrl+S → "Webpage, HTML only"** → save into `data/raw/`.
2. Run `python ingest.py` — it prints every scheme detail link it found.
3. Open those detail pages and save them into `data/raw/` too (the list page alone has little detail).
4. Run `python ingest.py` again.

### 2. Build the graph
```bash
python build_graph.py
```

### 3. Chat
```bash
streamlit run app.py          # web UI with chat + graph explorer
python retriever.py           # or quick terminal test
```

### Optional: Neo4j
Paste `data/graph.cypher` into Neo4j Browser (Aura free tier works), then try:
```cypher
MATCH (s:Scheme)-[:TARGETS]->(b:BeneficiaryGroup {name:'small and marginal farmers'}) RETURN s.name;
MATCH (s:Scheme)-[r]->(n) RETURN s, r, n LIMIT 100;
```

## Sample questions for the demo
1. List all schemes.
2. Which schemes are for small and marginal farmers?
3. What documents are needed for *<scheme name>*?
4. Which schemes give a subsidy for drip irrigation?
5. How do I apply for *<scheme name>*?
6. Something not in the data (to show it refuses to make things up).

## Files
| File | Purpose |
|---|---|
| `config.py` | Paths, model names, LLM/embedding factories |
| `ingest.py` | Parse saved HTML/PDF/TXT into clean text |
| `build_graph.py` | LLM extraction → knowledge graph, Cypher export, FAISS index |
| `retriever.py` | Hybrid graph + vector retrieval and answer generation |
| `app.py` | Streamlit chat UI with graph evidence and explorer |
