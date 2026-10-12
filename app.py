"""Step 4 - Streamlit chat UI.   Run:  streamlit run app.py"""
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from config import GRAPH_JSON

COLORS = {"Scheme": "#2563eb", "Department": "#7c3aed", "BeneficiaryGroup": "#059669",
          "Eligibility": "#d97706", "Benefit": "#16a34a", "Document": "#dc2626",
          "ApplicationChannel": "#0891b2", "Sector": "#65a30d", "Funding": "#9333ea"}

st.set_page_config(page_title="TN Schemes GraphRAG", page_icon="🌾", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_rag():
    from retriever import GraphRAG
    return GraphRAG()


def render_graph(g, height=520):
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff", cdn_resources="in_line")
    for n, d in g.nodes(data=True):
        label = d["label"] if len(d["label"]) < 40 else d["label"][:37] + "..."
        net.add_node(n, label=label, title=f"{d['type']}: {d['label']}",
                     color=COLORS.get(d["type"], "#64748b"), size=22 if d["type"] == "Scheme" else 12)
    for u, v, d in g.edges(data=True):
        net.add_edge(u, v, title=d["relation"], label=d["relation"], font={"size": 8})
    net.repulsion(node_distance=160, spring_length=140)
    components.html(net.generate_html(), height=height + 20)


if not GRAPH_JSON.exists():
    st.error("Graph not built yet. Run `python ingest.py` then `python build_graph.py`.")
    st.stop()

rag = load_rag()
g = rag.g
types = {}
for _, d in g.nodes(data=True):
    types[d["type"]] = types.get(d["type"], 0) + 1

with st.sidebar:
    st.header("🌾 TN Schemes GraphRAG")
    st.caption("Knowledge-graph RAG over Tamil Nadu government scheme pages")
    st.metric("Schemes", types.get("Scheme", 0))
    c1, c2 = st.columns(2)
    c1.metric("Nodes", g.number_of_nodes())
    c2.metric("Edges", g.number_of_edges())
    st.markdown("**Try asking**")
    examples = ["List all schemes",
                "Which schemes are for small and marginal farmers?",
                "What documents do I need to apply?",
                "Which schemes give subsidy for drip irrigation?"]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

tab_chat, tab_graph = st.tabs(["💬 Chat", "🕸️ Explore graph"])

with tab_chat:
    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("Ask about a scheme, eligibility, benefits, documents...") or st.session_state.pop("pending", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Walking the graph..."):
                out = rag.answer(question, history=st.session_state.messages[:-1])
            st.markdown(out["answer"])
            with st.expander(f"🔎 Graph evidence — {len(out['schemes'])} scheme(s), {len(out['facts'])} facts"):
                if out["entities"]:
                    st.write("**Matched entities:**", ", ".join(g.nodes[e]["label"] for e in out["entities"]))
                st.code("\n".join(out["facts"][:60]) or "(none)", language=None)
                if out["subgraph"].number_of_nodes():
                    render_graph(out["subgraph"], height=380)
        st.session_state.messages.append({"role": "assistant", "content": out["answer"]})

with tab_graph:
    schemes = sorted(d["label"] for _, d in g.nodes(data=True) if d["type"] == "Scheme")
    pick = st.multiselect("Show schemes (empty = whole graph)", schemes)
    if pick:
        ids = [rag.scheme_ids[p.lower()] for p in pick]
        nodes = set(ids) | {t for i in ids for t in g.successors(i)}
        render_graph(g.subgraph(nodes), height=620)
    else:
        render_graph(g, height=620)
    st.caption(" · ".join(f"{t}: {c}" for t, c in sorted(types.items())))
