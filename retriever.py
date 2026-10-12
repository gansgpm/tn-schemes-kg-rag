"""
Step 3 - GraphRAG retrieval + answer generation.

For each question:
  1. Vector search (FAISS)            -> seed schemes + supporting raw text
  2. Entity linking on graph nodes    -> e.g. "women farmers", "aadhaar", "paddy"
  3. Graph expansion                  -> all facts (triples) of the top schemes,
                                         and schemes connected to matched entities
  4. LLM answers from triples + text only, citing scheme names
"""
import json
import re
from collections import Counter

import networkx as nx
from rapidfuzz import fuzz

from config import FAISS_DIR, GRAPH_JSON, get_embeddings, get_llm

STOPWORDS = set("""a an the of for to in on and or is are what which who how can i my me do does
any all scheme schemes tamil nadu tn government govt under with by from get about there
list tell give show need required available""".split())

ANSWER_PROMPT = """You are a helpful assistant for Tamil Nadu government schemes.
Answer the question using ONLY the knowledge-graph facts and source text below.
- Name the scheme(s) you are drawing from.
- Use bullet points for eligibility, benefits, documents and steps.
- If the facts do not contain the answer, say so plainly and suggest contacting
  the department or the nearest office. Do not invent amounts or conditions.

KNOWLEDGE GRAPH FACTS:
{facts}

SOURCE TEXT:
{text}

CONVERSATION SO FAR:
{history}

QUESTION: {question}
ANSWER:"""


class GraphRAG:
    def __init__(self, llm=None, embeddings=None):
        data = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))
        self.g: nx.MultiDiGraph = nx.node_link_graph(data, directed=True, multigraph=True, edges="links")
        from langchain_community.vectorstores import FAISS
        self.vs = FAISS.load_local(str(FAISS_DIR), embeddings or get_embeddings(),
                                   allow_dangerous_deserialization=True)
        self.llm = llm or get_llm(temperature=0.2)
        self.scheme_ids = {d["label"].lower(): n for n, d in self.g.nodes(data=True) if d["type"] == "Scheme"}
        self.entity_nodes = [(n, d) for n, d in self.g.nodes(data=True) if d["type"] != "Scheme"]

    # ---------- retrieval ----------
    def link_entities(self, question: str, threshold: int = 85):
        """Match question words to non-scheme graph nodes (fuzzy)."""
        q = question.lower()
        terms = [w for w in re.findall(r"[a-z0-9]+", q) if w not in STOPWORDS and len(w) > 2]
        if not terms:
            return []
        q_clean = " ".join(terms)
        hits = []
        for n, d in self.entity_nodes:
            label = d["label"].lower()
            if len(label) > 80:          # long sentences (eligibility text) -> use token overlap
                score = fuzz.token_set_ratio(q_clean, label) if len(terms) > 1 else 0
            else:
                score = fuzz.partial_ratio(label, q_clean) if len(label) >= 4 else 0
            if score >= threshold:
                hits.append((n, score))
        return [n for n, _ in sorted(hits, key=lambda x: -x[1])[:10]]

    def link_schemes_by_name(self, question: str):
        q = question.lower()
        return [sid for name, sid in self.scheme_ids.items() if fuzz.partial_ratio(name, q) >= 90 and len(name) > 5]

    def retrieve(self, question: str, k: int = 6, max_schemes: int = 5):
        scores = Counter()

        # 1. vector search
        docs = self.vs.similarity_search(question, k=k)
        for rank, d in enumerate(docs):
            if d.metadata.get("kind") == "scheme":
                sid = self.scheme_ids.get(d.metadata["scheme"].lower())
                if sid:
                    scores[sid] += (k - rank)
        text_chunks = [d.page_content for d in docs if d.metadata.get("kind") == "chunk"][:3]

        # 2. explicit scheme names in the question get top priority
        for sid in self.link_schemes_by_name(question):
            scores[sid] += 20

        # 3. entity linking -> schemes connected to those entities (graph hop)
        entities = self.link_entities(question)
        for ent in entities:
            for sid in self.g.predecessors(ent):
                scores[sid] += 4

        list_all = bool(re.search(r"\b(all|list|how many|which schemes)\b", question.lower()))
        top = [sid for sid, _ in scores.most_common(15 if list_all and entities else max_schemes)]

        # 4. expand: collect triples for the chosen schemes
        facts, sub_nodes = [], set()
        for sid in top:
            s = self.g.nodes[sid]
            sub_nodes.add(sid)
            if s.get("description"):
                facts.append(f"({s['label']}) -[DESCRIPTION]-> {s['description']}")
            for _, tgt, ed in self.g.out_edges(sid, data=True):
                facts.append(f"({s['label']}) -[{ed['relation']}]-> ({self.g.nodes[tgt]['label']})")
                sub_nodes.add(tgt)

        if list_all and not entities:
            names = sorted(d["label"] for _, d in self.g.nodes(data=True) if d["type"] == "Scheme")
            facts.insert(0, f"The graph contains {len(names)} schemes: " + "; ".join(names))

        return {"facts": facts, "text": text_chunks, "schemes": top,
                "entities": entities, "subgraph": self.g.subgraph(sub_nodes).copy()}

    # ---------- generation ----------
    def answer(self, question: str, history=None):
        ctx = self.retrieve(question)
        hist = "\n".join(f"{m['role']}: {m['content']}" for m in (history or [])[-4:]) or "(none)"
        prompt = ANSWER_PROMPT.format(
            facts="\n".join(ctx["facts"]) or "(no matching facts)",
            text="\n---\n".join(ctx["text"]) or "(none)",
            history=hist, question=question)
        reply = self.llm.invoke(prompt)
        ctx["answer"] = getattr(reply, "content", str(reply))
        return ctx


if __name__ == "__main__":   # quick CLI test:  python retriever.py
    rag = GraphRAG()
    while True:
        q = input("\nAsk (blank to quit): ").strip()
        if not q:
            break
        out = rag.answer(q)
        print("\n" + out["answer"])
        print(f"\n[schemes used: {[rag.g.nodes[s]['label'] for s in out['schemes']]}]")
