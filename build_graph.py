"""
Step 2 - Build the Knowledge Graph (+ a vector index for hybrid retrieval).

  pages.json --LLM structured extraction--> schemes.json
  schemes.json --> NetworkX graph (graph.json) + Neo4j Cypher (graph.cypher)
  schemes + raw chunks --> FAISS index

Graph schema
  Nodes : Scheme, Department, BeneficiaryGroup, Eligibility, Benefit,
          Document, ApplicationChannel, Sector, Funding
  Edges : Scheme -ADMINISTERED_BY-> Department
          Scheme -TARGETS->         BeneficiaryGroup
          Scheme -HAS_ELIGIBILITY-> Eligibility
          Scheme -PROVIDES->        Benefit
          Scheme -REQUIRES->        Document
          Scheme -APPLY_VIA->       ApplicationChannel
          Scheme -COVERS->          Sector
          Scheme -FUNDED_BY->       Funding
Shared nodes (e.g. "small and marginal farmers", "aadhaar card") link
schemes together - that is what lets the chatbot answer cross-scheme
questions a plain vector RAG struggles with.
"""
import json
import re
from typing import List, Optional

import networkx as nx
from pydantic import BaseModel, Field

from config import (CHUNK_OVERLAP, CHUNK_SIZE, CYPHER_FILE, FAISS_DIR, GRAPH_JSON,
                    PAGES_JSON, SCHEMES_JSON, get_embeddings, get_llm)


# ---------- 1. Extraction schema ----------
class SchemeInfo(BaseModel):
    name: str = Field(description="Official scheme name as written in the text")
    department: Optional[str] = Field(None, description="Implementing department")
    description: Optional[str] = Field(None, description="1-3 sentence summary of the objective")
    beneficiaries: List[str] = Field(default_factory=list, description="Target groups, e.g. 'small and marginal farmers'")
    eligibility: List[str] = Field(default_factory=list, description="Each eligibility condition as a short phrase")
    benefits: List[str] = Field(default_factory=list, description="Each benefit/assistance, keep amounts and % exactly")
    documents_required: List[str] = Field(default_factory=list)
    how_to_apply: List[str] = Field(default_factory=list, description="Offices, officers, portals or apps to apply through")
    sectors: List[str] = Field(default_factory=list, description="Crops, activities or sectors covered")
    funding: Optional[str] = Field(None, description="Funding pattern, e.g. 'State scheme', 'Centrally sponsored 60:40'")


class Extraction(BaseModel):
    schemes: List[SchemeInfo] = Field(default_factory=list)


EXTRACTION_PROMPT = """You are building a knowledge graph of Tamil Nadu government schemes.
Extract every scheme described in the text below.

Rules:
- Use ONLY information stated in the text. Never guess or add outside knowledge.
- Leave a field empty if the text does not mention it.
- List items must be short noun phrases (max ~12 words), one fact per item.
- Keep rupee amounts, percentages and limits exactly as written.
- Use consistent wording for common things: "aadhaar card", "small and marginal farmers",
  "agricultural extension centre", "uzhavan app".

TEXT (from {source}):
{text}"""


def chunk(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):                      # break on a paragraph if possible
            cut = text.rfind("\n", start + size // 2, end)
            end = cut if cut > 0 else end
        out.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return out


def extract_from_text(llm, text: str, source: str) -> List[SchemeInfo]:
    structured = llm.with_structured_output(Extraction)
    result = structured.invoke(EXTRACTION_PROMPT.format(text=text, source=source))
    return result.schemes if result else []


# ---------- 2. Normalisation & merge ----------
SYNONYMS = {
    "aadhar": "aadhaar", "aadhar card": "aadhaar card", "aadhaar": "aadhaar card",
    "chitta / adangal": "chitta and adangal", "chitta/adangal": "chitta and adangal",
    "small & marginal farmers": "small and marginal farmers",
    "uzhavan mobile app": "uzhavan app", "uzhavan application": "uzhavan app",
}


def norm(label: str) -> str:
    s = re.sub(r"\s+", " ", (label or "").strip().strip(".;,:")).lower()
    return SYNONYMS.get(s, s)


def merge_schemes(items: List[SchemeInfo]) -> dict:
    merged = {}
    list_fields = ["beneficiaries", "eligibility", "benefits", "documents_required", "how_to_apply", "sectors"]
    for s in items:
        key = norm(s.name)
        if not key:
            continue
        cur = merged.setdefault(key, {"name": s.name.strip(), **{f: [] for f in list_fields},
                                      "department": None, "description": None, "funding": None})
        for f in list_fields:
            for v in getattr(s, f):
                if v and norm(v) not in {norm(x) for x in cur[f]}:
                    cur[f].append(v.strip())
        for f in ["department", "funding"]:
            cur[f] = cur[f] or getattr(s, f)
        if s.description and len(s.description) > len(cur["description"] or ""):
            cur["description"] = s.description
    return merged


# ---------- 3. Graph ----------
RELATIONS = [  # field, relation, target node type
    ("beneficiaries", "TARGETS", "BeneficiaryGroup"),
    ("eligibility", "HAS_ELIGIBILITY", "Eligibility"),
    ("benefits", "PROVIDES", "Benefit"),
    ("documents_required", "REQUIRES", "Document"),
    ("how_to_apply", "APPLY_VIA", "ApplicationChannel"),
    ("sectors", "COVERS", "Sector"),
]


def node_id(ntype: str, label: str) -> str:
    return f"{ntype}::{norm(label)}"


def build_graph(schemes: dict) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    def add_edge(sid, ntype, label, rel):
        tid = node_id(ntype, label)
        if not g.has_node(tid):
            g.add_node(tid, type=ntype, label=label.strip())
        g.add_edge(sid, tid, relation=rel)

    for s in schemes.values():
        sid = node_id("Scheme", s["name"])
        g.add_node(sid, type="Scheme", label=s["name"], description=s.get("description") or "")
        if s.get("department"):
            add_edge(sid, "Department", s["department"], "ADMINISTERED_BY")
        if s.get("funding"):
            add_edge(sid, "Funding", s["funding"], "FUNDED_BY")
        for field, rel, ntype in RELATIONS:
            for v in s.get(field, []):
                add_edge(sid, ntype, v, rel)
    return g


def scheme_profile(s: dict) -> str:
    """Readable text block for one scheme (used for embeddings and context)."""
    lines = [f"Scheme: {s['name']}"]
    for label, key in [("Department", "department"), ("Funding", "funding"), ("Description", "description")]:
        if s.get(key):
            lines.append(f"{label}: {s[key]}")
    for label, key in [("Beneficiaries", "beneficiaries"), ("Eligibility", "eligibility"),
                       ("Benefits", "benefits"), ("Documents required", "documents_required"),
                       ("How to apply", "how_to_apply"), ("Sectors", "sectors")]:
        if s.get(key):
            lines.append(f"{label}: " + "; ".join(s[key]))
    return "\n".join(lines)


def export_cypher(g: nx.MultiDiGraph, path):
    """Optional: load into Neo4j (Aura/Desktop) by pasting into the browser or cypher-shell."""
    esc = lambda x: str(x).replace("\\", "\\\\").replace("'", "\\'")
    out = [f"MERGE (:{d['type']} {{id:'{esc(n)}', name:'{esc(d['label'])}'}});" for n, d in g.nodes(data=True)]
    for u, v, d in g.edges(data=True):
        out.append(f"MATCH (a {{id:'{esc(u)}'}}), (b {{id:'{esc(v)}'}}) MERGE (a)-[:{d['relation']}]->(b);")
    path.write_text("\n".join(out), encoding="utf-8")


# ---------- 4. Vector index ----------
def build_vector_index(schemes: dict, pages: list):
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    docs = [Document(page_content=scheme_profile(s),
                     metadata={"kind": "scheme", "scheme": s["name"]}) for s in schemes.values()]
    for p in pages:
        for i, c in enumerate(chunk(p["text"], 1200, 150)):
            docs.append(Document(page_content=c, metadata={"kind": "chunk", "source": p["source"], "chunk": i}))
    FAISS.from_documents(docs, get_embeddings()).save_local(str(FAISS_DIR))
    return len(docs)


def main():
    pages = json.loads(PAGES_JSON.read_text(encoding="utf-8"))
    llm = get_llm()

    extracted = []
    for p in pages:
        for i, c in enumerate(chunk(p["text"])):
            found = extract_from_text(llm, c, p["source"])
            print(f"  {p['source']} chunk {i}: {len(found)} scheme(s)")
            extracted.extend(found)

    schemes = merge_schemes(extracted)
    SCHEMES_JSON.write_text(json.dumps(list(schemes.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    g = build_graph(schemes)
    GRAPH_JSON.write_text(json.dumps(nx.node_link_data(g, edges="links"), ensure_ascii=False), encoding="utf-8")
    export_cypher(g, CYPHER_FILE)
    n_docs = build_vector_index(schemes, pages)

    print(f"\nSchemes: {len(schemes)} | Nodes: {g.number_of_nodes()} | Edges: {g.number_of_edges()} | Vectors: {n_docs}")
    print(f"Saved {SCHEMES_JSON.name}, {GRAPH_JSON.name}, {CYPHER_FILE.name}, {FAISS_DIR.name}/")


if __name__ == "__main__":
    main()
