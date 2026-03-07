import json
import logging
import sys
from pathlib import Path

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingestion import load_knowledge_base, ingest_new_document
from src.embeddings import EmbeddingIndex
from src.llm_client import OllamaClient
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)



@st.cache_resource(show_spinner=False)
def get_index(config: dict) -> EmbeddingIndex:
    return EmbeddingIndex(config)


@st.cache_resource(show_spinner=False)
def get_llm(config: dict) -> OllamaClient:
    return OllamaClient(config)


def get_pipeline(config: dict) -> RAGPipeline:
    index = get_index(config)
    llm   = get_llm(config)
    return RAGPipeline(config, index, llm)



def _inject_css() -> None:
    st.markdown("""
    <style>
    /* ── Page & font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Header banner ── */
    .nust-header {
        background: linear-gradient(135deg, #003580 0%, #0057b8 100%);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .nust-header h1 {
        color: white !important;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
    }
    .nust-header p {
        color: rgba(255,255,255,0.85);
        font-size: 0.9rem;
        margin: 0;
    }

    /* ── Chat bubbles ── */
    .chat-row { display: flex; margin-bottom: 16px; gap: 10px; }
    .chat-row.user  { flex-direction: row-reverse; }
    .chat-row.bot   { flex-direction: row; }

    .avatar {
        width: 38px; height: 38px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .avatar.user { background: #0057b8; }
    .avatar.bot  { background: #003580; }

    .bubble {
        max-width: 75%;
        padding: 12px 16px;
        border-radius: 16px;
        font-size: 0.92rem;
        line-height: 1.55;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .bubble.user {
        background: #0057b8;
        color: white;
        border-bottom-right-radius: 4px;
    }
    .bubble.bot {
        background: #f0f4ff;
        color: #1a1a2e;
        border-bottom-left-radius: 4px;
    }

    /* ── Sources pill ── */
    .source-pill {
        display: inline-block;
        background: #e8f0fe;
        color: #003580;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 10px;
        margin: 2px 3px;
    }

    /* ── Status badges ── */
    .badge-green { color: #1a7f37; font-weight: 600; }
    .badge-red   { color: #cf1322; font-weight: 600; }
    .badge-yellow { color: #b45309; font-weight: 600; }

    /* ── Input area ── */
    .stTextInput > div > div > input {
        border-radius: 24px;
        border: 2px solid #0057b8;
        padding: 10px 18px;
        font-size: 0.95rem;
    }
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 0 3px rgba(0,87,184,0.2);
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 24px;
        background: #0057b8;
        color: white;
        font-weight: 600;
        border: none;
        padding: 8px 20px;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #003580; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0d1b3e;
    }
    section[data-testid="stSidebar"] * {
        color: #e8edf7 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] strong {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #c8d4ee !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #8fa3c8 !important;
    }
    /* Sidebar input fields */
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: #1a2d5a;
        border: 1.5px solid #3a5a9a;
        color: #e8edf7 !important;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] .stTextInput > div > div > input::placeholder {
        color: #6b84b0 !important;
    }
    section[data-testid="stSidebar"] .stTextInput label {
        color: #c8d4ee !important;
    }
    /* Sidebar file uploader */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #1a2d5a;
        border: 1.5px dashed #3a5a9a;
        border-radius: 8px;
        padding: 8px;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
        color: #c8d4ee !important;
    }
    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        background: #1651a8;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        width: 100%;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #0057b8;
    }
    /* Sidebar divider */
    section[data-testid="stSidebar"] hr {
        border-color: #2a3f6f;
    }
    /* Badge colours */
    section[data-testid="stSidebar"] .badge-green { color: #4ade80 !important; }
    section[data-testid="stSidebar"] .badge-red   { color: #f87171 !important; }
    section[data-testid="stSidebar"] .badge-yellow { color: #fbbf24 !important; }
    /* Inline code blocks in sidebar */
    section[data-testid="stSidebar"] code {
        background: #1a2d5a !important;
        color: #93c5fd !important;
        border-radius: 4px;
        padding: 1px 5px;
    }

    /* ── Divider ── */
    hr { border-color: #dce6f5; margin: 16px 0; }
    </style>
    """, unsafe_allow_html=True)



def _render_header(cfg: dict) -> None:
    app_cfg = cfg.get("app", {})
    st.markdown(f"""
    <div class="nust-header">
        <div>NUST Bank</div>
        <div>
            <h1>{app_cfg.get('title', 'NUST Bank AI Assistant')}</h1>
            <p>{app_cfg.get('subtitle', 'Your intelligent banking support partner')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_message(role: str, content: str, sources: list | None = None) -> None:
    icon = "User" if role == "user" else "Bot"
    st.markdown(f"""
    <div class="chat-row {role}">
        <div class="avatar {role}">{icon}</div>
        <div class="bubble {role}">{content}</div>
    </div>
    """, unsafe_allow_html=True)

    if sources:
        cats = list(dict.fromkeys(r["category"] for r in sources if r.get("category")))[:4]
        pills = "".join(f'<span class="source-pill">📂 {c}</span>' for c in cats)
        st.markdown(f'<div style="margin-left:48px;margin-top:-8px;">{pills}</div>',
                    unsafe_allow_html=True)


def _render_sources_expander(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"View {len(sources)} source(s) used", expanded=False):
        for i, src in enumerate(sources, 1):
            score_pct = int(src.get("score", 0) * 100)
            st.markdown(
                f"**{i}. {src.get('category', 'Unknown')}**  "
                f"<span style='color:#888;font-size:0.8rem;'>relevance: {score_pct}%</span>",
                unsafe_allow_html=True,
            )
            if src.get("question"):
                st.markdown(f"*Q: {src['question']}*")
            st.markdown(src.get("answer", "")[:400] +
                        ("…" if len(src.get("answer", "")) > 400 else ""))
            if i < len(sources):
                st.markdown("---")


def _render_sidebar(cfg: dict, index: EmbeddingIndex, llm: OllamaClient) -> None:
    with st.sidebar:
        st.markdown("## System Status")

        # LLM status
        if llm.is_available():
            st.markdown(f"<span class='badge-green'>● LLM Online</span> — `{cfg['llm']['model']}`",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='badge-red'>● LLM Offline</span> — `{cfg['llm']['model']}`",
                        unsafe_allow_html=True)
            st.warning(
                f"Ollama not running or model not found.\n\n"
                f"**Fix:**\n```\nollama serve\nollama pull {cfg['llm']['model']}\n```"
            )

        # Index status
        stats = index.get_stats()
        if "error" not in stats:
            n = stats.get("vectors_count", 0)
            st.markdown(
                f"<span class='badge-green'>● Index Ready</span> — {n:,} documents",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<span class='badge-red'>● Index Not Built</span>", unsafe_allow_html=True)
            st.warning("Run `python ingest.py` to build the index.")

        st.markdown("---")

        # ── Document Upload ──
        st.markdown("## Add New Document")
        st.caption("Upload a new FAQ or policy document to expand the knowledge base instantly.")

        upload_category = st.text_input(
            "Product / Category name",
            placeholder="e.g. NUST Student Loan",
            key="upload_category",
        )
        uploaded_file = st.file_uploader(
            "Upload .txt or .json file",
            type=["txt", "json"],
            key="uploaded_file",
        )

        if st.button("Add to Knowledge Base", key="btn_upload"):
            if not uploaded_file:
                st.error("Please select a file to upload.")
            elif not upload_category.strip():
                st.error("Please enter a category name.")
            else:
                with st.spinner("Processing document..."):
                    try:
                        raw_text = uploaded_file.read().decode("utf-8", errors="replace")

                        # If JSON, try to extract text content
                        if uploaded_file.name.endswith(".json"):
                            try:
                                data = json.loads(raw_text)
                                if isinstance(data, dict):
                                    raw_text = json.dumps(data, indent=2)
                                elif isinstance(data, list):
                                    raw_text = "\n".join(str(item) for item in data)
                            except json.JSONDecodeError:
                                pass  # treat as plain text

                        new_doc = ingest_new_document(raw_text, upload_category.strip(), cfg)
                        index.add([new_doc])

                        st.success(f"Document added! (id: {new_doc['id']})")
                        st.caption("You can now ask questions about this new content.")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

        st.markdown("---")

        # ── Rebuild Index ──
        st.markdown("## Rebuild Index")
        st.caption("Re-process source files and rebuild the full vector index.")
        if st.button("Rebuild from Source Files", key="btn_rebuild"):
            with st.spinner("Re-ingesting & re-indexing (this may take ~1 minute)..."):
                try:
                    from src.data_ingestion import run_ingestion
                    docs = run_ingestion(cfg)
                    get_index.clear()
                    fresh_index = EmbeddingIndex(cfg)
                    fresh_index.build(docs)
                    # Invalidate cached index
                    st.cache_resource.clear()
                    st.success(f"✅ Index rebuilt with {len(docs)} documents.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

        st.markdown("---")

        # ── Contact ──
        app_cfg = cfg.get("app", {})
        st.markdown("## Contact NUST Bank")
        st.markdown(
            f"**Helpline:** {app_cfg.get('helpline', '+92 (51) 111 000 494')}  \n"
            f"**Email:** {app_cfg.get('support_email', 'support@NUSTbank.com.pk')}"
        )

        # ── Clear chat ──
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", key="btn_clear"):
            st.session_state.messages = []
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="NUST Bank AI Assistant",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_css()

    # Load config & resources
    cfg      = load_config()
    index    = get_index(cfg)
    llm      = get_llm(cfg)
    pipeline = get_pipeline(cfg)

    # Ensure index is built
    if not index.collection_exists():
        st.warning(
            "The knowledge base index has not been built yet.  \n"
            "Run **`python ingest.py`** in your terminal to ingest the bank data and build the index.",
            icon="⚠️",
        )

    # Sidebar
    _render_sidebar(cfg, index, llm)

    # Header
    _render_header(cfg)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Greeting
        st.session_state.messages.append({
            "role": "bot",
            "content": (
                "Hello! I'm the **NUST Bank AI Assistant**.\n\n"
                "I can help you with information about our accounts, loans, credit cards, "
                "digital banking services, home remittance, and more.\n\n"
                "How can I assist you today?"
            ),
            "sources": [],
        })

    # Render history
    for msg in st.session_state.messages:
        _render_message(msg["role"], msg["content"], msg.get("sources"))
        if msg["role"] == "bot" and msg.get("sources"):
            _render_sources_expander(msg["sources"])

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Suggested questions ──
    if len(st.session_state.messages) <= 1:
        st.markdown("**💡 Suggested questions:**")
        suggestions = [
            "What is the NUST Asaan Account?",
            "How do I transfer funds using the mobile app?",
            "What are the features of the NUST Waqaar Account?",
            "Can I apply for auto finance from NUST Bank?",
            "What are the profit rates on savings accounts?",
        ]
        cols = st.columns(len(suggestions))
        for col, suggestion in zip(cols, suggestions):
            if col.button(suggestion, key=f"sug_{suggestion[:20]}"):
                st.session_state["prefill_query"] = suggestion
                st.rerun()

    # ── Input form ──
    prefill = st.session_state.pop("prefill_query", "")

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                label="Ask a question",
                value=prefill,
                placeholder="e.g. What documents do I need to open a current account?",
                label_visibility="collapsed",
                key="chat_input",
            )
        with col2:
            submitted = st.form_submit_button("Send 📨")

    # ── Handle submission ──
    if submitted and user_input.strip():
        query = user_input.strip()

        # Add user message
        st.session_state.messages.append({"role": "user", "content": query, "sources": []})
        _render_message("user", query)

        # Stream bot response
        bot_placeholder = st.empty()
        full_response = ""
        sources: list[dict] = []

        with st.spinner(""):
            for token in pipeline.stream_query(query):
                if token.startswith("\n__SOURCES__:"):
                    sources_json = token[len("\n__SOURCES__:"):]
                    try:
                        sources = json.loads(sources_json)
                    except Exception:
                        sources = []
                else:
                    full_response += token
                    bot_placeholder.markdown(
                        f'<div class="chat-row bot">'
                        f'<div class="avatar bot">🤖</div>'
                        f'<div class="bubble bot">{full_response}▌</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Final render (remove cursor)
        bot_placeholder.empty()
        _render_message("bot", full_response, sources)
        if sources:
            _render_sources_expander(sources)

        # Persist
        st.session_state.messages.append({
            "role": "bot",
            "content": full_response,
            "sources": sources,
        })

        st.rerun()


if __name__ == "__main__":
    main()
