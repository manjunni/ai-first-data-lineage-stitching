from abc import ABC, abstractmethod
from src.graph.models import LineageFragment


class BaseParser(ABC):
    """All artifact parsers implement this interface."""

    @abstractmethod
    def can_parse(self, artifact: dict) -> bool:
        """Return True if this parser handles the given artifact."""

    @abstractmethod
    def parse(self, artifact: dict) -> LineageFragment:
        """Extract lineage fragment from the artifact."""
