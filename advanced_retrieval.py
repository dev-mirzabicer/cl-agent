# Advanced Retrieval Tools for Combinatory Logic RAG

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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


@dataclass
class KGSearchResult:
    """Structured result from a knowledge graph search."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class KnowledgeGraphTool:
    """
    A tool for the agent to interact with the knowledge graph.
    Provides a flexible search interface that the agent can configure on the fly.
    """

    def __init__(self, system):
        self.system = system
        self.graphiti = system.graphiti

    async def search(
        self,
        query: str,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        search_methods: Optional[List[NodeSearchMethod]] = None,
        limit: int = 20,
        group_ids: Optional[List[str]] = None,
    ) -> KGSearchResult:
        """
        Performs a flexible search on the knowledge graph.

        Args:
            query: The search query.
            node_types: Optional list of node labels to search for (e.g., ["Fact", "Definition"]).
            edge_types: Optional list of edge labels to search for.
            search_methods: Optional list of search methods to use.
            limit: The maximum number of results to return.
            group_ids: Optional list of group IDs to search within.

        Returns:
            A structured search result containing nodes and edges.
        """
        print(
            f"🔎 Searching KG for: '{query}' with config: node_types={node_types}, limit={limit}"
        )

        # Default search methods if not provided
        if not search_methods:
            search_methods = [NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity]

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

        search_results = await self.graphiti.search_(
            query=query,
            config=config,
            group_ids=group_ids
            if group_ids
            else ["combinatory-logic-books", "combinatory-logic-papers"],
        )

        # Format results into simple dicts for the agent
        nodes = [node.dict() for node in search_results.nodes]
        edges = [edge.dict() for edge in search_results.edges]

        print(f"✅ Found {len(nodes)} nodes and {len(edges)} edges.")
        return KGSearchResult(nodes=nodes, edges=edges)

    async def close(self):
        """Close any resources opened by the tool."""
        pass  # graphiti handles its own connections


@dataclass
class SourceReadResult:
    """Structured result from a direct source read."""

    source_id: str
    content: str
    metadata: Dict[str, Any]


class DirectSourceReaderTool:
    """
    A tool for the agent to read full source documents or sections directly.
    """

    def __init__(self, system):
        self.system = system
        self.neo4j_driver = self.system.graphiti.driver

    async def read_source(
        self, source_id: str, section_query: Optional[str] = None
    ) -> SourceReadResult:
        """
        Reads the full text of a source or a specific section within it.

        Args:
            source_id: The unique identifier for the source document.
            section_query: Optional query to find a specific section (e.g., "Chapter 3").

        Returns:
            The content of the source or section.
        """
        print(f"📖 Reading source: {source_id}, section: {section_query}")

        if section_query:
            # Find the specific section (episode) within the source
            query = """
            MATCH (s:Source {id: $source_id})<-[:PART_OF]-(e:Episodic)
            WHERE e.name CONTAINS $section_query
            RETURN e.content as content, e.name as name
            LIMIT 1
            """
            params = {"source_id": source_id, "section_query": section_query}
        else:
            # Return the entire document content (if stored as a single episode)
            query = """
            MATCH (s:Source {id: $source_id})<-[:PART_OF]-(e:Episodic)
            RETURN s.full_text as content, s.title as name
            LIMIT 1
            """
            params = {"source_id": source_id}

        async with self.neo4j_driver.session() as session:
            result = await session.run(query, params)
            record = await result.single()

            if record:
                content = record["content"]
                name = record["name"]
                print(f"✅ Found source content: {name} ({len(content)} chars)")
                return SourceReadResult(
                    source_id=source_id, content=content, metadata={"source_name": name}
                )
            else:
                print(f"❌ Source or section not found: {source_id}, {section_query}")
                return SourceReadResult(
                    source_id=source_id,
                    content="Error: Source or section not found.",
                    metadata={},
                )

    async def close(self):
        """Close any resources opened by the tool."""
        pass  # neo4j driver is managed by graphiti
