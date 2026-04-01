from abc import ABC, abstractmethod
from typing import Any


class Connector(ABC):
    """
    Abstract base class for knowledge graph storage connectors.
    """

    def __init__(self, graph_id: str | None = None):
        """
        Initializes the connector.

        Args:
            graph_id: Unique identifier for the knowledge graph (for isolation)
        """
        self.graph_id = graph_id

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @abstractmethod
    def close(self):
        """Closes the connection to the storage."""
        pass

    @abstractmethod
    def upsert_entities(self, entities: list) -> None:
        """
        Creates or updates nodes in the storage for each entity in the list.
        """
        pass

    @abstractmethod
    def upsert_relations(self, relations: list) -> None:
        """
        Creates or updates relationships in the storage for each relation in the list.
        """
        pass

    @abstractmethod
    def get_all_entities(self) -> list:
        """
        Retrieves all entities from the storage.
        """
        pass

    @abstractmethod
    def get_all_relations(self) -> list:
        """
        Retrieves all relations from the storage.
        """
        pass

    @abstractmethod
    def run_query(self, query: str, parameters: dict | None = None) -> Any:
        """
        Runs a native query against the storage.
        """
        pass

    @abstractmethod
    def clear(self, confirm: str | None = None) -> None:
        """
        Clears all data from the storage.
        """
        pass

    def _confirm_clear(self, confirm: str | None = None) -> bool:
        """
        Standard confirmation logic for clearing data.
        Returns True if the operation should proceed, False otherwise.
        """
        graph_id = self.graph_id or "default"
        if confirm == "yes":
            return True

        user_input = input(f"确定要清除图 '{graph_id}' 的所有数据吗？此操作不可恢复！(y/n): ")
        if user_input.lower() == "y":
            return True

        print("操作已取消。")
        return False
