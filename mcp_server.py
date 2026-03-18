import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_disk_server")

# Check if config.toml exists
CONFIG_PATH = os.environ.get("DISK_CONFIG_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.toml"
)
if not os.path.exists(CONFIG_PATH):
    logger.error(
        f"config.toml not found at {CONFIG_PATH}. "
        + "Please create it from config.example.toml before running the MCP server."
    )
    # We'll still initialize FastMCP so the server starts
    # but tools will fail with informative messages.
    CONFIG_EXISTS = False
else:
    # If using custom path, update the config module's path
    import config.llm

    config.llm.CONFIG_PATH = CONFIG_PATH
    CONFIG_EXISTS = True

# Initialize FastMCP
mcp = FastMCP("DISK Knowledge Graph Server")

# Initialize DISK instance lazily to avoid crash if config is missing
_disk_instance = None


def get_disk():
    global _disk_instance
    if not CONFIG_EXISTS:
        raise FileNotFoundError("config.toml not found. Please create it from config.example.toml.")
    if _disk_instance is None:
        from config.llm import embeddings, llm
        from disk import DISK

        _disk_instance = DISK(llm=llm, embeddings=embeddings)
    return _disk_instance


@mcp.tool()
def distill_document(file_path: str) -> list[str]:
    """
    Distill a document (PDF/DOCX) into text blocks.

    Args:
        file_path: The absolute path to the document file.

    Returns:
        A list of extracted text blocks.
    """
    if not os.path.exists(file_path):
        return [f"Error: File not found at {file_path}"]

    try:
        disk = get_disk()
        blocks = disk.distiller.extract_text_blocks(file_path)
        return blocks
    except Exception as e:
        logger.error(f"Error distilling document: {e}")
        return [f"Error: {str(e)}"]


@mcp.tool()
def extract_knowledge_from_text(text: str, pdf_path: str | None = None) -> dict[str, Any]:
    """
    Extract entities and relations from a single text block using LLM.

    Args:
        text: The text content to analyze.
        pdf_path: Optional path to the PDF file for context/language detection.

    Returns:
        A dictionary containing lists of 'entities' and 'relations'.
    """
    try:
        disk = get_disk()
        result = disk.extractor.extract_relations_and_entities(text, pdf_path=pdf_path)
        if result is None:
            return {"entities": [], "relations": []}

        relations, entities = result
        return {
            "entities": [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in entities],
            "relations": [r.to_dict() if hasattr(r, "to_dict") else str(r) for r in relations],
        }
    except Exception as e:
        logger.error(f"Error extracting knowledge: {e}")
        return {"error": str(e)}


@mcp.tool()
def merge_knowledge_fragments(
    entities1: list[dict], relations1: list[dict], entities2: list[dict], relations2: list[dict]
) -> dict[str, Any]:
    """
    Merge two sets of knowledge fragments (entities and relations) by resolving duplicates semantically.

    Args:
        entities1: First list of entities.
        relations1: First list of relations.
        entities2: Second list of entities.
        relations2: Second list of relations.

    Returns:
        A dictionary with merged 'entities' and 'relations'.
    """
    try:
        # Convert dicts back to objects if necessary
        # Assuming Merger works with lists of objects that have certain attributes
        # We need to see what Entity/Relation objects look like to reconstruct them
        # For now, we'll try to use the raw dicts if the merger supports it,
        # or we might need a helper to reconstruct models.

        # Simple reconstruction (this might need refinement based on models/__init__.py)
        from models import Entity, Relation

        def dict_to_entity(d):
            if isinstance(d, dict):
                # Ensure all required fields are present
                return Entity(
                    label=d.get("label", ""),
                    name=d.get("name", ""),
                    description=d.get("description", ""),
                    embedding=None,  # In MCP we typically don't pass embeddings back and forth
                )
            return d

        def dict_to_relation(d):
            if isinstance(d, dict):
                return Relation(
                    start_entity=dict_to_entity(d.get("start_entity")),
                    end_entity=dict_to_entity(d.get("end_entity")),
                    label=d.get("label", ""),
                    name=d.get("name", ""),
                    description=d.get("description", ""),
                    embedding=None,
                )
            return d

        e1 = [dict_to_entity(e) for e in entities1]
        r1 = [dict_to_relation(r) for r in relations1]
        e2 = [dict_to_entity(e) for e in entities2]
        r2 = [dict_to_relation(r) for r in relations2]

        disk = get_disk()
        merged_relations, merged_entities = disk.merger.merge(
            entities1=e1, relations1=r1, entities2=e2, relations2=r2
        )

        return {
            "entities": [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in merged_entities],
            "relations": [
                r.to_dict() if hasattr(r, "to_dict") else str(r) for r in merged_relations
            ],
        }
    except Exception as e:
        logger.error(f"Error merging knowledge: {e}")
        return {"error": str(e)}


@mcp.tool()
def build_full_knowledge_graph(file_path: str, mode: str = "parallel") -> dict[str, Any]:
    """
    Run the full pipeline to build a knowledge graph from a document.

    Args:
        file_path: Absolute path to the PDF/DOCX file.
        mode: Execution mode, "parallel" or "serial".

    Returns:
        The resulting knowledge graph as a dictionary.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        disk = get_disk()
        kg = disk.build_knowledge_graph(
            file_path, mode="parallel" if mode == "parallel" else "serial"
        )

        return {
            "entities": [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in kg.entities],
            "relations": [r.to_dict() if hasattr(r, "to_dict") else str(r) for r in kg.relations],
        }
    except Exception as e:
        logger.error(f"Error building KG: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
