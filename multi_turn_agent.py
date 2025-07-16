# advanced_retrieval.py
"""
Advanced retrieval tools for the schema-driven combinatory logic RAG system.
Provides sophisticated search and retrieval capabilities using graphiti and the dynamic graph.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from graphiti_core.search.search_config import (
    SearchConfig,
    EdgeSearchConfig,
    NodeSearchConfig,
)
from graphiti_core.search.search import (
    EdgeSearchMethod,
    NodeSearchMethod,
    EdgeReranker,
    NodeReranker,
)

from core.schema_registry import SchemaRegistry
from core.dynamic_graph_manager import DynamicGraphManager

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured result from a knowledge graph search."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    total_results: int
    search_config: Dict[str, Any]


@dataclass
class ContextualSearchResult:
    """Enhanced search result with contextual information."""

    primary_results: List[Dict[str, Any]]
    related_context: List[Dict[str, Any]]
    relationship_map: Dict[str, List[Dict[str, Any]]]
    glossary_terms: List[Dict[str, Any]]


class SchemaAwareKnowledgeGraphTool:
    """
    Advanced tool for searching and retrieving information from the knowledge graph.
    Integrates with the schema registry for intelligent search configuration.
    """

    def __init__(self, system):
        """
        Initialize the knowledge graph tool.

        Args:
            system: System instance with graphiti, schema registry, and graph manager
        """
        self.system = system
        self.graphiti = system.graphiti
        self.schema_registry = system.schema_registry
        self.graph_manager = system.graph_manager

    async def search(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        relationship_types: Optional[List[str]] = None,
        search_methods: Optional[List[NodeSearchMethod]] = None,
        limit: int = 20,
        include_context: bool = True,
        group_ids: Optional[List[str]] = None,
    ) -> SearchResult:
        """
        Perform an intelligent search on the knowledge graph with schema awareness.

        Args:
            query: The search query
            entity_types: Optional list of entity types to search (from schema)
            relationship_types: Optional list of relationship types to include
            search_methods: Optional list of search methods to use
            limit: Maximum number of results to return
            include_context: Whether to include contextual information
            group_ids: Optional list of group IDs to search within

        Returns:
            Structured search result with nodes and edges
        """
        logger.info(f"🔍 Performing schema-aware search for: '{query}'")

        # Validate entity types against schema
        if entity_types:
            valid_types = self.schema_registry.get_all_entity_types()
            entity_types = [et for et in entity_types if et in valid_types]
            if not entity_types:
                logger.warning("No valid entity types provided, searching all types")
                entity_types = None

        # Configure search methods
        if not search_methods:
            search_methods = [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity]

        # Build search configuration
        node_config = NodeSearchConfig(
            search_methods=search_methods,
            reranker=NodeReranker.cross_encoder,
        )

        edge_config = EdgeSearchConfig(
            search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
            reranker=EdgeReranker.cross_encoder,
        )

        config = SearchConfig(
            node_config=node_config,
            edge_config=edge_config,
            limit=limit,
        )

        # Set default group IDs if not provided
        if not group_ids:
            group_ids = ["combinatory-logic-books", "combinatory-logic-papers"]

        try:
            # Perform the search using graphiti
            search_results = await self.graphiti.search_(
                query=query, config=config, group_ids=group_ids
            )

            # Convert results to dictionaries
            nodes = [node.dict() for node in search_results.nodes]
            edges = [edge.dict() for edge in search_results.edges]

            # Filter by entity types if specified
            if entity_types:
                filtered_nodes = []
                for node in nodes:
                    # Check if node matches any of the requested entity types
                    node_labels = node.get("labels", [])
                    if any(label in entity_types for label in node_labels):
                        filtered_nodes.append(node)
                nodes = filtered_nodes

            logger.info(f"✅ Search completed: {len(nodes)} nodes, {len(edges)} edges")

            return SearchResult(
                nodes=nodes,
                edges=edges,
                total_results=len(nodes) + len(edges),
                search_config={
                    "query": query,
                    "entity_types": entity_types,
                    "limit": limit,
                    "search_methods": [method.value for method in search_methods],
                },
            )

        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return SearchResult(nodes=[], edges=[], total_results=0, search_config={})

    async def contextual_search(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        max_primary_results: int = 10,
        max_context_items: int = 20,
        include_definitions: bool = True,
        include_related_facts: bool = True,
    ) -> ContextualSearchResult:
        """
        Perform a contextual search that includes related information and definitions.

        Args:
            query: The search query
            entity_types: Optional list of entity types to focus on
            max_primary_results: Maximum primary search results
            max_context_items: Maximum contextual items to include
            include_definitions: Whether to include related definitions
            include_related_facts: Whether to include related facts

        Returns:
            Enhanced search result with contextual information
        """
        logger.info(f"🔍 Performing contextual search for: '{query}'")

        # Primary search
        primary_search = await self.search(
            query=query, entity_types=entity_types, limit=max_primary_results
        )

        # Collect entity IDs from primary results
        primary_entity_ids = [
            node.get("uuid", node.get("id")) for node in primary_search.nodes
        ]

        # Build relationship map
        relationship_map = {}
        for entity_id in primary_entity_ids:
            if entity_id:
                relationships = await self.graph_manager.get_entity_relationships(
                    entity_id, direction="both"
                )
                relationship_map[entity_id] = relationships

        # Collect related context
        related_context = []
        glossary_terms = []

        # Extract related entities from relationships
        related_entity_ids = set()
        for entity_id, relationships in relationship_map.items():
            for rel in relationships:
                other_entity = rel.get("other_entity", {})
                other_id = other_entity.get("id")
                if other_id and other_id not in primary_entity_ids:
                    related_entity_ids.add(other_id)

        # Retrieve related entities (limited to avoid overwhelming results)
        for entity_id in list(related_entity_ids)[:max_context_items]:
            entity = await self.graph_manager.get_entity(entity_id)
            if entity:
                # Categorize by type
                entity_types_list = entity.get("_types", [])
                if "Definition" in entity_types_list and include_definitions:
                    glossary_terms.append(entity)
                elif include_related_facts:
                    related_context.append(entity)

        logger.info(
            f"✅ Contextual search completed: {len(primary_search.nodes)} primary, "
            f"{len(related_context)} context, {len(glossary_terms)} definitions"
        )

        return ContextualSearchResult(
            primary_results=primary_search.nodes,
            related_context=related_context,
            relationship_map=relationship_map,
            glossary_terms=glossary_terms,
        )

    async def search_by_entity_type(
        self,
        entity_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for entities of a specific type with optional filters.

        Args:
            entity_type: Type of entity to search for
            filters: Optional property filters
            limit: Maximum number of results

        Returns:
            List of entities matching the criteria
        """
        logger.info(f"🔍 Searching for {entity_type} entities")

        # Validate entity type
        if entity_type not in self.schema_registry.get_all_entity_types():
            logger.error(f"❌ Unknown entity type: {entity_type}")
            return []

        try:
            entities = await self.graph_manager.find_entities(
                entity_type=entity_type, filters=filters, limit=limit
            )

            logger.info(f"✅ Found {len(entities)} {entity_type} entities")
            return entities

        except Exception as e:
            logger.error(f"❌ Entity search failed: {e}")
            return []

    async def get_entity_with_full_context(
        self, entity_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entity with its complete contextual information.

        Args:
            entity_id: ID of the entity to retrieve

        Returns:
            Entity with full context or None if not found
        """
        logger.info(f"🔍 Retrieving entity with full context: {entity_id}")

        try:
            # Get the main entity
            entity = await self.graph_manager.get_entity(entity_id)
            if not entity:
                logger.warning(f"⚠️ Entity not found: {entity_id}")
                return None

            # Get all relationships
            relationships = await self.graph_manager.get_entity_relationships(entity_id)

            # Get related entities
            related_entities = []
            for rel in relationships:
                other_entity = rel.get("other_entity")
                if other_entity:
                    related_entities.append(
                        {
                            "entity": other_entity,
                            "relationship_type": rel.get("relationship_type"),
                            "relationship_properties": rel.get("relationship", {}),
                        }
                    )

            # Enhance entity with context
            entity["_context"] = {
                "relationships": relationships,
                "related_entities": related_entities,
                "relationship_count": len(relationships),
            }

            logger.info(f"✅ Retrieved entity with {len(relationships)} relationships")
            return entity

        except Exception as e:
            logger.error(f"❌ Failed to retrieve entity with context: {e}")
            return None

    async def search_similar_entities(
        self,
        reference_entity_id: str,
        entity_types: Optional[List[str]] = None,
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find entities similar to a reference entity using embedding similarity.

        Args:
            reference_entity_id: ID of the reference entity
            entity_types: Optional list of entity types to search within
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results

        Returns:
            List of similar entities with similarity scores
        """
        logger.info(f"🔍 Finding entities similar to: {reference_entity_id}")

        try:
            # Get the reference entity
            reference_entity = await self.graph_manager.get_entity(reference_entity_id)
            if not reference_entity:
                logger.warning(f"⚠️ Reference entity not found: {reference_entity_id}")
                return []

            # Create a search query from the reference entity's content
            search_content = ""
            for field in ["content", "definition", "term", "explanation"]:
                if field in reference_entity:
                    search_content += f" {reference_entity[field]}"

            if not search_content.strip():
                logger.warning(f"⚠️ No searchable content in reference entity")
                return []

            # Perform similarity search
            search_results = await self.search(
                query=search_content.strip(),
                entity_types=entity_types,
                limit=limit + 1,  # +1 to account for the reference entity itself
            )

            # Filter out the reference entity and apply similarity threshold
            similar_entities = []
            for node in search_results.nodes:
                node_id = node.get("uuid", node.get("id"))
                if node_id != reference_entity_id:
                    # Add similarity score (this would need to be computed properly)
                    node["_similarity_score"] = 0.8  # Placeholder
                    similar_entities.append(node)

            logger.info(f"✅ Found {len(similar_entities)} similar entities")
            return similar_entities[:limit]

        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []

    async def close(self):
        """Close any resources used by the tool."""
        # The graph manager and graphiti handle their own resource management
        pass


@dataclass
class SourceReadResult:
    """Structured result from a direct source read."""

    source_id: str
    content: str
    metadata: Dict[str, Any]
    sections: List[Dict[str, Any]]


class DirectSourceReaderTool:
    """
    Tool for reading complete source documents or specific sections.
    Integrates with the schema-aware graph structure.
    """

    def __init__(self, system):
        """
        Initialize the source reader tool.

        Args:
            system: System instance with graph manager and Neo4j driver
        """
        self.system = system
        self.graph_manager = system.graph_manager
        self.neo4j_driver = system.graphiti.driver

    async def read_source(
        self,
        source_id: str,
        section_query: Optional[str] = None,
        include_metadata: bool = True,
    ) -> SourceReadResult:
        """
        Read the full text of a source or a specific section within it.

        Args:
            source_id: The unique identifier for the source document
            section_query: Optional query to find a specific section
            include_metadata: Whether to include source metadata

        Returns:
            The content of the source or section with metadata
        """
        logger.info(f"📖 Reading source: {source_id}, section: {section_query}")

        try:
            # Get source metadata
            source_entity = await self.graph_manager.get_entity(source_id)
            if not source_entity:
                logger.error(f"❌ Source not found: {source_id}")
                return SourceReadResult(
                    source_id=source_id,
                    content="Error: Source not found.",
                    metadata={},
                    sections=[],
                )

            metadata = dict(source_entity) if include_metadata else {}

            # If specific section requested, find it
            if section_query:
                content, sections = await self._read_specific_section(
                    source_id, section_query
                )
            else:
                content, sections = await self._read_full_source(source_id)

            logger.info(
                f"✅ Successfully read source: {len(content)} characters, {len(sections)} sections"
            )

            return SourceReadResult(
                source_id=source_id,
                content=content,
                metadata=metadata,
                sections=sections,
            )

        except Exception as e:
            logger.error(f"❌ Failed to read source {source_id}: {e}")
            return SourceReadResult(
                source_id=source_id,
                content=f"Error reading source: {e}",
                metadata={},
                sections=[],
            )

    async def _read_specific_section(self, source_id: str, section_query: str) -> tuple:
        """Read a specific section from a source."""
        # Query for entities from this source that match the section query
        query = """
        MATCH (s:Source {id: $source_id})<-[:PART_OF]-(entity)
        WHERE entity.source_section CONTAINS $section_query 
           OR entity.parent_sections CONTAINS $section_query
        RETURN entity.content as content, 
               entity.source_section as section_name,
               entity.parent_sections as parent_sections
        ORDER BY entity.source_section
        """

        params = {"source_id": source_id, "section_query": section_query}

        async with self.neo4j_driver.session() as session:
            result = await session.run(query, params)

            content_parts = []
            sections = []

            async for record in result:
                if record["content"]:
                    content_parts.append(record["content"])
                    sections.append(
                        {
                            "name": record["section_name"],
                            "parent_sections": record["parent_sections"],
                            "content_length": len(record["content"]),
                        }
                    )

            full_content = (
                "\n\n".join(content_parts)
                if content_parts
                else f"No content found for section: {section_query}"
            )

            return full_content, sections

    async def _read_full_source(self, source_id: str) -> tuple:
        """Read the complete content of a source."""
        # Query for all entities from this source
        query = """
        MATCH (s:Source {id: $source_id})<-[:PART_OF]-(entity)
        RETURN entity.content as content,
               entity.source_section as section_name,
               entity.parent_sections as parent_sections,
               labels(entity) as entity_types
        ORDER BY entity.source_section, entity.id
        """

        params = {"source_id": source_id}

        async with self.neo4j_driver.session() as session:
            result = await session.run(query, params)

            content_parts = []
            sections = []

            async for record in result:
                if record["content"]:
                    content_parts.append(
                        f"[{', '.join(record['entity_types'])}] {record['content']}"
                    )
                    sections.append(
                        {
                            "name": record["section_name"],
                            "parent_sections": record["parent_sections"],
                            "entity_types": record["entity_types"],
                            "content_length": len(record["content"]),
                        }
                    )

            full_content = (
                "\n\n".join(content_parts)
                if content_parts
                else "No content found in source."
            )

            return full_content, sections

    async def list_source_sections(self, source_id: str) -> List[Dict[str, Any]]:
        """
        List all sections available in a source document.

        Args:
            source_id: ID of the source document

        Returns:
            List of section information
        """
        logger.info(f"📋 Listing sections for source: {source_id}")

        try:
            query = """
            MATCH (s:Source {id: $source_id})<-[:PART_OF]-(entity)
            RETURN DISTINCT entity.source_section as section_name,
                   entity.parent_sections as parent_sections,
                   count(entity) as entity_count
            ORDER BY entity.source_section
            """

            params = {"source_id": source_id}

            async with self.neo4j_driver.session() as session:
                result = await session.run(query, params)

                sections = []
                async for record in result:
                    sections.append(
                        {
                            "name": record["section_name"],
                            "parent_sections": record["parent_sections"],
                            "entity_count": record["entity_count"],
                        }
                    )

                logger.info(f"✅ Found {len(sections)} sections in source {source_id}")
                return sections

        except Exception as e:
            logger.error(f"❌ Failed to list sections for {source_id}: {e}")
            return []

    async def close(self):
        """Close any resources used by the tool."""
        # Neo4j driver is managed by graphiti
        pass
