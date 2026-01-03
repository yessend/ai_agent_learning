import json

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.vector_stores.qdrant import QdrantVectorStore
from .logger import logger

class DualSchemaQdrantVectorStore(QdrantVectorStore):
    """
    Custom QdrantVectorStore that handles multiple document schemas.
    For documents with doc_type='point', it uses the 'text' field.
    For documents with doc_type='table', it uses 'table_data' converted to string.
    """

    def parse_to_query_result(self, response):
        """
        Override parse_to_query_result to handle dual schema ('text' and '_node_content').

        Args:
            response: The search result response from the Qdrant client,
                      expected to be an iterable of ScoredPoint objects.

        Returns:
            VectorStoreQueryResult: Parsed query result object for LlamaIndex.
        """
        nodes = []
        ids = []
        similarities = []

        for point in response:
            point_id_str = str(point.id) # Store ID for logger/use
            payload = point.payload or {}
            embedding = point.vector if hasattr(point, "vector") else None

            # --- Metadata Extraction ---
            # Use parent method for consistency if possible, otherwise basic extraction
            try:
                # Note: _extract_metadata might be protected; direct access is safer
                # metadata = payload.get("_node_metadata", {}) # Example default LlamaIndex key
                # node_info = payload.get("_node_info", {})
                # relationships = payload.get("_node_relationships", {})

                # Safer: Extract known metadata, let TextNode handle relationships if needed
                metadata = {}
                # relationships = {}
                # node_info = {} # For start/end char, often not critical if not present
                for key, value in payload.items():
                     # Exclude known non-metadata keys or internal keys if necessary
                     if key not in ["text", "_node_content", "table_data", "_node_type", "_node_info", "_node_metadata", "_node_relationships"]:
                          metadata[key] = value
                     # Optionally, merge _node_metadata if it exists
                     if key == "_node_metadata" and isinstance(value, dict):
                          metadata.update(value)


            except Exception as meta_err:
                logger.error(f"Error extracting metadata for point {point_id_str}: {meta_err}", exc_info=True)
                metadata = payload.get("metadata", {}) # Basic fallback

            # --- Text Content Extraction ---
            doc_type = payload.get("doc_type", "point") # Default to 'point' if missing
            text_content = "" # Default to empty string

            if doc_type == "table":
                # logger.debug(f"Processing point {point_id_str} as doc_type 'table'")
                table_data = payload.get("table_data", {})
                if table_data:
                    try:
                        # Convert table data (likely dict/list) to JSON string
                        text_content = json.dumps(table_data, ensure_ascii=False, indent=2)
                    except TypeError:
                        logger.warning(f"Could not JSON dump table_data for point {point_id_str}: {table_data}. Using str().")
                        text_content = str(table_data) # Fallback to simple string conversion
                else:
                    logger.warning(f"doc_type is 'table' but 'table_data' key missing/empty in payload for point {point_id_str}")

            else: # Handle 'point' or any other doc_type
                # logger.debug(f"Processing point {point_id_str} as doc_type '{doc_type}' (or default)")
                # 1. Try the 'text' key first (compatible with your manual indexing)
                text_content = payload.get("summary")

                # 2. If 'text' is missing or empty/whitespace, try '_node_content' (compatible with default LlamaIndex)
                if not text_content or (isinstance(text_content, str) and text_content.isspace()):
                    # logger.debug(f"'text' key missing or empty for point {point_id_str}. Trying '_node_content'.")
                    text_content = payload.get("_node_content")

                    # 3. If both are missing or empty/whitespace, log a warning
                    # if not text_content or (isinstance(text_content, str) and text_content.isspace()):
                    #     logger.warning(f"Both 'text' and '_node_content' keys are missing or empty in payload for point {point_id_str}. Node content will be empty.")
                    #     text_content = "" # Ensure it's an empty string

            # Final check: Ensure text_content is a string before passing to TextNode
            if not isinstance(text_content, str):
                logger.warning(f"Payload text content is not a string (type: {type(text_content)}) for point {point_id_str}. Converting to string.")
                text_content = str(text_content)


            # --- Create TextNode ---
            try:
                node = TextNode(
                    id_=point_id_str,
                    text=text_content, # Use the determined text_content
                    metadata=metadata, # Pass extracted metadata
                    # LlamaIndex can reconstruct relationships if keys exist in metadata/payload
                    # start_char_idx=node_info.get("start", None),
                    # end_char_idx=node_info.get("end", None),
                    # relationships=relationships, # Or let TextNode handle from metadata
                    embedding=embedding, # Pass embedding if available
                )
                nodes.append(node)
                ids.append(point_id_str)

                # Extract similarity score if available
                if hasattr(point, "score"):
                    similarities.append(point.score)
                else:
                     # Handle cases where score might be missing (though unlikely in search)
                     similarities.append(None) # Or maybe 0.0?

            except Exception as node_err:
                 logger.error(f"Failed to create TextNode for point {point_id_str}: {node_err}", exc_info=True)


        # Filter out potential None scores if your downstream code expects floats
        valid_similarities = [s for s in similarities if s is not None]
        if len(valid_similarities) != len(nodes):
             logger.warning("Some retrieved points were missing similarity scores.")
             # Decide how to handle - either pad with 0.0 or ensure calling code handles None
             # For now, we pass the list possibly containing None values if needed.
             # Or adjust: valid_similarities = [s if s is not None else 0.0 for s in similarities]


        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=valid_similarities if len(valid_similarities) == len(nodes) else similarities)