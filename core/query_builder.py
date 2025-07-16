# core/query_builder.py
"""
Dynamic Cypher query builder that generates queries based on schema configuration.
Provides type-safe query generation with automatic parameter handling.
"""

from typing import Dict, List, Tuple, Any, Optional
import logging
from core.schema_registry import SchemaRegistry
from models.schema_models import EntitySchema, RelationshipSchema

logger = logging.getLogger(__name__)


class CypherQueryBuilder:
    """
    Builds Cypher queries dynamically based on schema definitions.
    Ensures type safety and proper parameter handling.
    """

    def __init__(self, schema_registry: SchemaRegistry):
        """
        Initialize the query builder.

        Args:
            schema_registry: Registry containing schema definitions
        """
        self.schema = schema_registry

    def build_create_entity_query(
        self, entity_type: str, data: Dict[str, Any], source_id: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a Cypher query to create an entity with its properties.

        Args:
            entity_type: Type of entity to create
            data: Entity data dictionary
            source_id: Optional source document ID

        Returns:
            Tuple of (cypher_query, parameters)
        """
        entity_schema = self.schema.get_entity_schema(entity_type)
        if not entity_schema:
            raise ValueError(f"Unknown entity type: {entity_type}")

        # Validate required ID
        if "id" not in data:
            raise ValueError(f"Entity data must include 'id' field")

        # Build property setters dynamically
        property_setters = []
        params = {"entity_id": data["id"]}

        for prop_name, prop_value in data.items():
            if prop_name in entity_schema.properties:
                param_name = f"prop_{prop_name}"
                property_setters.append(f"e.{prop_name} = ${param_name}")
                params[param_name] = prop_value

        # Build the query
        query_parts = [
            f"MERGE (e:{entity_type} {{id: $entity_id}})",
            f"SET {', '.join(property_setters)}"
            if property_setters
            else "// No properties to set",
        ]

        # Add source relationship if provided
        if source_id:
            query_parts.extend(
                [
                    "WITH e",
                    "MATCH (s:Source {id: $source_id})",
                    "MERGE (e)-[:PART_OF]->(s)",
                ]
            )
            params["source_id"] = source_id

        # Add return statement
        query_parts.append("RETURN e.id as created_id")

        query = "\n".join(query_parts)
        logger.debug(f"Generated create entity query for {entity_type}: {query}")

        return query, params

    def build_create_relationship_query(
        self,
        relationship_type: str,
        source_id: str,
        target_id: str,
        properties: Dict[str, Any] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a Cypher query to create a relationship between entities.

        Args:
            relationship_type: Type of relationship to create
            source_id: ID of the source entity
            target_id: ID of the target entity
            properties: Optional relationship properties

        Returns:
            Tuple of (cypher_query, parameters)
        """
        rel_schema = self.schema.get_relationship_schema(relationship_type)
        if not rel_schema:
            raise ValueError(f"Unknown relationship type: {relationship_type}")

        params = {"source_id": source_id, "target_id": target_id}

        # Build relationship properties if provided
        rel_properties = []
        if properties:
            for prop_name, prop_value in properties.items():
                if prop_name in rel_schema.properties:
                    param_name = f"rel_{prop_name}"
                    rel_properties.append(f"{prop_name}: ${param_name}")
                    params[param_name] = prop_value

        # Build the relationship creation part
        rel_props_str = "{" + ", ".join(rel_properties) + "}" if rel_properties else ""

        query = f"""
        MATCH (source {{id: $source_id}})
        MATCH (target {{id: $target_id}})
        MERGE (source)-[r:{relationship_type} {rel_props_str}]->(target)
        RETURN r
        """

        logger.debug(
            f"Generated create relationship query for {relationship_type}: {query}"
        )

        return query, params

    def build_find_entity_query(
        self,
        entity_type: str,
        filters: Dict[str, Any] = None,
        limit: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a query to find entities by type and optional filters.

        Args:
            entity_type: Type of entity to find
            filters: Optional property filters
            limit: Optional result limit

        Returns:
            Tuple of (cypher_query, parameters)
        """
        entity_schema = self.schema.get_entity_schema(entity_type)
        if not entity_schema:
            raise ValueError(f"Unknown entity type: {entity_type}")

        params = {}
        where_conditions = []

        # Build WHERE conditions from filters
        if filters:
            for prop_name, prop_value in filters.items():
                if prop_name in entity_schema.properties:
                    param_name = f"filter_{prop_name}"
                    where_conditions.append(f"e.{prop_name} = ${param_name}")
                    params[param_name] = prop_value

        # Build the query
        query_parts = [f"MATCH (e:{entity_type})"]

        if where_conditions:
            query_parts.append(f"WHERE {' AND '.join(where_conditions)}")

        query_parts.append("RETURN e")

        if limit:
            query_parts.append(f"LIMIT {limit}")

        query = "\n".join(query_parts)
        logger.debug(f"Generated find entity query for {entity_type}: {query}")

        return query, params

    def build_get_entity_relationships_query(
        self,
        entity_id: str,
        relationship_types: List[str] = None,
        direction: str = "both",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a query to get relationships for an entity.

        Args:
            entity_id: ID of the entity
            relationship_types: Optional list of relationship types to filter
            direction: "outgoing", "incoming", or "both"

        Returns:
            Tuple of (cypher_query, parameters)
        """
        params = {"entity_id": entity_id}

        # Build relationship type filter
        rel_filter = ""
        if relationship_types:
            rel_types_str = "|".join(relationship_types)
            rel_filter = f":{rel_types_str}"

        # Build direction-specific patterns
        if direction == "outgoing":
            pattern = f"(e)-[r{rel_filter}]->(other)"
        elif direction == "incoming":
            pattern = f"(e)<-[r{rel_filter}]-(other)"
        else:  # both
            pattern = f"(e)-[r{rel_filter}]-(other)"

        query = f"""
        MATCH (e {{id: $entity_id}})
        MATCH {pattern}
        RETURN r, other
        """

        logger.debug(
            f"Generated get relationships query for entity {entity_id}: {query}"
        )

        return query, params

    def build_delete_entity_query(self, entity_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Build a query to delete an entity and all its relationships.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            Tuple of (cypher_query, parameters)
        """
        params = {"entity_id": entity_id}

        query = """
        MATCH (e {id: $entity_id})
        DETACH DELETE e
        """

        logger.debug(f"Generated delete entity query for {entity_id}: {query}")

        return query, params

    def build_update_entity_query(
        self, entity_id: str, updates: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a query to update entity properties.

        Args:
            entity_id: ID of the entity to update
            updates: Dictionary of property updates

        Returns:
            Tuple of (cypher_query, parameters)
        """
        params = {"entity_id": entity_id}

        # Build SET clauses for updates
        set_clauses = []
        for prop_name, prop_value in updates.items():
            param_name = f"update_{prop_name}"
            set_clauses.append(f"e.{prop_name} = ${param_name}")
            params[param_name] = prop_value

        if not set_clauses:
            raise ValueError("No valid updates provided")

        query = f"""
        MATCH (e {{id: $entity_id}})
        SET {", ".join(set_clauses)}
        RETURN e
        """

        logger.debug(f"Generated update entity query for {entity_id}: {query}")

        return query, params

    def build_fulltext_search_query(
        self, entity_types: List[str], query_text: str, limit: int = 50
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a full-text search query across entity types.

        Args:
            entity_types: List of entity types to search
            query_text: Search query text
            limit: Maximum number of results

        Returns:
            Tuple of (cypher_query, parameters)
        """
        params = {"query_text": query_text, "limit": limit}

        # Build UNION queries for each entity type
        union_queries = []
        for entity_type in entity_types:
            index_name = f"{entity_type.lower()}_content_index"
            union_queries.append(f"""
                CALL db.index.fulltext.queryNodes('{index_name}', $query_text) 
                YIELD node, score
                RETURN node, score, '{entity_type}' as entity_type
            """)

        query = " UNION ALL ".join(union_queries)
        query += f"\nORDER BY score DESC LIMIT $limit"

        logger.debug(f"Generated fulltext search query for {entity_types}: {query}")

        return query, params

    def build_schema_constraint_queries(self) -> List[str]:
        """
        Build queries to create schema constraints from the registry.

        Returns:
            List of Cypher constraint queries
        """
        setup_commands = self.schema.get_neo4j_setup_commands()
        return setup_commands.get("constraints", [])

    def build_schema_index_queries(self) -> List[str]:
        """
        Build queries to create schema indexes from the registry.

        Returns:
            List of Cypher index queries
        """
        setup_commands = self.schema.get_neo4j_setup_commands()
        return setup_commands.get("indexes", [])

    def build_get_entity_type_query(self, entity_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Build a query to get the type(s) of an entity by its ID.

        Args:
            entity_id: ID of the entity

        Returns:
            Tuple of (cypher_query, parameters)
        """
        params = {"entity_id": entity_id}

        query = """
        MATCH (e {id: $entity_id})
        RETURN labels(e) as entity_types
        """

        logger.debug(f"Generated get entity type query for {entity_id}: {query}")

        return query, params

    def build_count_entities_query(
        self, entity_type: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a query to count entities, optionally by type.

        Args:
            entity_type: Optional entity type to count

        Returns:
            Tuple of (cypher_query, parameters)
        """
        params = {}

        if entity_type:
            query = f"MATCH (e:{entity_type}) RETURN count(e) as count"
        else:
            query = "MATCH (e) RETURN count(e) as count"

        logger.debug(f"Generated count entities query: {query}")

        return query, params

    def build_get_glossary_query(
        self, source_id: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a query to retrieve definitions for glossary building.

        Args:
            source_id: Optional source ID to filter by

        Returns:
            Tuple of (cypher_query, parameters)
        """
        params = {}

        if source_id:
            query = """
            MATCH (d:Definition)-[:PART_OF]->(s:Source {id: $source_id})
            RETURN d.term as term, d.definition as definition, 'Definition' as type
            ORDER BY d.term
            """
            params["source_id"] = source_id
        else:
            query = """
            MATCH (d:Definition)
            RETURN d.term as term, d.definition as definition, 'Definition' as type
            ORDER BY d.term
            """

        logger.debug(f"Generated glossary query: {query}")

        return query, params
