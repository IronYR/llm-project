import json
import logging
from typing import Any, Generator, TypedDict

from langgraph.graph import END, START, StateGraph

from .guardrails import check_input, check_output, is_banking_related
from .prompts import (
    NO_INFO_RESPONSE,
    OFF_TOPIC_RESPONSE,
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
)

logger = logging.getLogger(__name__)


class RAGState(TypedDict, total=False):
    query: str
    guard_blocked: bool
    guard_message: str
    guard_warning: str
    results: list
    route: str
    answer: str
    sources: list


class RAGGraphRunner:
    def __init__(self, config: dict, index, llm) -> None:
        self.config = config
        self.index = index
        self.llm = llm
        self._graph = self._compile()

    def _retrieve(self, query: str) -> list[dict]:
        return self.index.search(query)

    def _is_off_topic(self, query: str, results: list[dict]) -> bool:
        has_kw = is_banking_related(query, self.config)
        return not has_kw and not results

    def _node_input_guard(self, state: RAGState) -> RAGState:
        gr = check_input(state["query"], self.config)
        if not gr.allowed:
            return {
                "guard_blocked": True,
                "guard_message": gr.safe_message,
                "answer": gr.safe_message,
                "sources": [],
            }
        return {"guard_blocked": False, "guard_warning": gr.warning or ""}

    def _route_after_input(self, state: RAGState) -> str:
        if state.get("guard_blocked"):
            return "end"
        return "retrieve"

    def _node_retrieve(self, state: RAGState) -> RAGState:
        q = state["query"]
        results = self._retrieve(q)
        if self._is_off_topic(q, results):
            return {
                "route": "off",
                "results": [],
                "answer": OFF_TOPIC_RESPONSE,
                "sources": [],
            }
        if not results:
            return {
                "route": "no_info",
                "results": [],
                "answer": NO_INFO_RESPONSE,
                "sources": [],
            }
        return {"route": "ok", "results": results}

    def _route_after_retrieve(self, state: RAGState) -> str:
        r = state.get("route")
        if r == "ok":
            return "generate"
        return "end"

    def _node_generate(self, state: RAGState) -> RAGState:
        results = state.get("results") or []
        ctx = build_context(results)
        up = build_user_prompt(state["query"], ctx)
        raw = self.llm.generate(SYSTEM_PROMPT, up)
        out = check_output(raw)
        ans = out.safe_message or raw
        return {"answer": ans, "sources": results}

    def _compile(self):
        g = StateGraph(RAGState)
        g.add_node("input_guard", self._node_input_guard)
        g.add_node("retrieve", self._node_retrieve)
        g.add_node("generate", self._node_generate)

        g.add_edge(START, "input_guard")
        g.add_conditional_edges(
            "input_guard",
            self._route_after_input,
            {"end": END, "retrieve": "retrieve"},
        )
        g.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"generate": "generate", "end": END},
        )
        g.add_edge("generate", END)
        return g.compile()

    def invoke(self, query: str) -> dict[str, Any]:
        out = self._graph.invoke({"query": query})
        gr = check_input(query, self.config)
        warn = out.get("guard_warning") or (gr.warning if gr.allowed else "")
        blocked = bool(out.get("guard_blocked")) or (
            out.get("answer") == OFF_TOPIC_RESPONSE
        )
        return {
            "answer": out.get("answer", ""),
            "sources": out.get("sources") or [],
            "blocked": blocked,
            "warning": warn or "",
        }

    def stream(self, query: str) -> Generator[str, None, None]:
        st: RAGState = {"query": query}
        st.update(self._node_input_guard(st))
        if st.get("guard_blocked"):
            yield st.get("answer", "")
            return

        gr = check_input(query, self.config)
        if gr.warning:
            yield f"*{gr.warning}*\n\n"

        st.update(self._node_retrieve(st))
        if st.get("route") != "ok":
            yield st.get("answer", "")
            return

        ctx = build_context(st.get("results") or [])
        up = build_user_prompt(query, ctx)
        full = ""
        for tok in self.llm.stream_generate(SYSTEM_PROMPT, up):
            full += tok
            yield tok

        out = check_output(full)
        if out.reason == "pii_sanitized":
            yield "\n\n[Response sanitized for privacy]"

        sources_payload = json.dumps(st.get("results") or [])
        yield f"\n__SOURCES__:{sources_payload}"
