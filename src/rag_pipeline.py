import logging
from typing import Generator

from .rag_graph import RAGGraphRunner

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: dict, index, llm) -> None:
        self.config = config
        self.index = index
        self.llm = llm
        self._runner = RAGGraphRunner(config, index, llm)

    def query(self, user_query: str) -> dict:
        return self._runner.invoke(user_query)

    def stream_query(self, user_query: str) -> Generator[str, None, None]:
        yield from self._runner.stream(user_query)

