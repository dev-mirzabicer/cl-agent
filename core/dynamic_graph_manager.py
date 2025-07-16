# core/dynamic_graph_manager.py
"""
Dynamic graph manager that provides schema-driven graph operations.
Handles entity and relationship creation, validation, and querying.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging
import uuid
from datetime import datetime

from core.schema_registry import SchemaRegistry
from core.query_builder import CypherQueryBuilder
from models.schema_models import ValidationResult, EntityData, RelationshipData

logger = logging.getLogger(__name__)


class GraphOperationError(Exception):
    """Custom exception for graph operation failures."""

    def __init__(self, operation: str, entity_type: str, details: str):
        self.operation = operation
        self.entity_type = entity_type
        self.details = details
        super().__init__(
            f"Graph operation failed: {operation} on {entity_type}: {details}"
        )


class DynamicGraphManager:
    """
    Schema-driven graph manager that provides high-level operations
    for creating, updating, and querying entities and relationships.
    """

    def __init__(self, neo4j_driver, schema_registry: SchemaRegistry):
        """
        Initialize the dynamic graph manager.

        Args:
            neo4j_driver: Neo4j driver instance
            schema_registry: Registry containing schema definitions
        """
        self.driver = neo4j_driver
        self.schema = schema_registry
        self.query_builder = CypherQueryBuilder(schema_registry)

    async def setup_database_schema(self) -> None:
        """Create database constraints and indexes based on schema configuration."""
        logger.info("🔧 Setting up database schema...")

        async with self.driver.session() as session:
            # Create constraints
            constraint_queries = self.query_builder.build_schema_constraint_queries()
            for query in constraint_queries:
                try:
                    await session.run(query)
                    logger.debug(f"✅ Created constraint: {query}")
                except Exception as e:
                    logger.warning(
                        f"⚠️ Constraint creation failed (may already exist): {e}"
                    )

            # Create indexes
            index_queries = self.query_builder.build_schema_index_queries()
            for query in index_queries:
                try:
                    await session.run(query)
                    logger.debug(f"✅ Created index: {query}")
                except Exception as e:
                    logger.warning(f"⚠️ Index creation failed (may already exist): {e}")

        logger.info("✅ Database schema setup complete")

    async def add_entity(
        self,
        entity_type: str,
        data: Dict[str, Any],
        source_id: Optional[str] = None,
        validate: bool = True,
    ) -> str:
        """
        Add a new entity to the graph with validation.

        Args:
            entity_type: Type of entity to create
            data: Entity data dictionary
            source_id: Optional source document ID
            validate: Whether to validate data against schema

        Returns:
            ID of the created entity

        Raises:
            GraphOperationError: If creation fails
        """
        try:
            # Generate ID if not provided
            if "id" not in data:
                data["id"] = self._generate_entity_id(entity_type)

            # Validate against schema if requested
            if validate:
                validation_result = self.schema.validate_entity_data(entity_type, data)
                if not validation_result.is_valid:
                    raise GraphOperationError(
                        "add_entity",
                        entity_type,
                        f"Validation failed: {', '.join(validation_result.errors)}",
                    )

            # Build and execute query
            query, params = self.query_builder.build_create_entity_query(
                entity_type, data, source_id
            )

            async with self.driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()

                if record:
                    created_id = record["created_id"]
                    logger.info(f"✅ Created {entity_type}: {created_id}")
                    return created_id
                else:
                    raise GraphOperationError(
                        "add_entity", entity_type, "No result returned from query"
                    )

        except Exception as e:
            logger.error(f"❌ Failed to create {entity_type}: {e}")
            if isinstance(e, GraphOperationError):
                raise
            raise GraphOperationError("add_entity", entity_type, str(e))

    async def create_relationship(
        self,
        relationship_type: str,
        source_id: str,
        target_id: str,
        properties: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> None:
        """
        Create a relationship between two entities.

        Args:
            relationship_type: Type of relationship to create
            source_id: ID of the source entity
            target_id: ID of the target entity
            properties: Optional relationship properties
            validate: Whether to validate the relationship

        Raises:
            GraphOperationError: If creation fails
        """
        try:
            # Validate relationship if requested
            if validate:
                source_entity_type = await self.get_entity_type(source_id)
                target_entity_type = await self.get_entity_type(target_id)

                validation_result = self.schema.validate_relationship_data(
                    relationship_type,
                    source_entity_type,
                    target_entity_type,
                    properties,
                )

                if not validation_result.is_valid:
                    raise GraphOperationError(
                        "create_relationship",
                        relationship_type,
                        f"Validation failed: {', '.join(validation_result.errors)}",
                    )

            # Build and execute query
            query, params = self.query_builder.build_create_relationship_query(
                relationship_type, source_id, target_id, properties or {}
            )

            async with self.driver.session() as session:
                await session.run(query, params)
                logger.info(
                    f"✅ Created relationship {relationship_type}: {source_id} -> {target_id}"
                )

        except Exception as e:
            logger.error(f"❌ Failed to create relationship {relationship_type}: {e}")
            if isinstance(e, GraphOperationError):
                raise
            raise GraphOperationError("create_relationship", relationship_type, str(e))

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: ID of the entity to retrieve

        Returns:
            Entity data dictionary or None if not found
        """
        try:
            query = "MATCH (e {id: $entity_id}) RETURN e, labels(e) as types"
            params = {"entity_id": entity_id}

            async with self.driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()

                if record:
                    entity_node = record["e"]
                    entity_types = record["types"]

                    entity_data = dict(entity_node)
                    entity_data["_types"] = entity_types

                    return entity_data

                return None

        except Exception as e:
            logger.error(f"❌ Failed to get entity {entity_id}: {e}")
            return None

    async def get_entity_type(self, entity_id: str) -> Optional[str]:
        """
        Get the primary entity type for an entity.

        Args:
            entity_id: ID of the entity

        Returns:
            Primary entity type or None if not found
        """
        try:
            query, params = self.query_builder.build_get_entity_type_query(entity_id)

            async with self.driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()

                if record and record["entity_types"]:
                    # Return the first type (entities should have one primary type)
                    return record["entity_types"][0]

                return None

        except Exception as e:
            logger.error(f"❌ Failed to get entity type for {entity_id}: {e}")
            return None

    async def find_entities(
        self,
        entity_type: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find entities by type and optional filters.

        Args:
            entity_type: Type of entities to find
            filters: Optional property filters
            limit: Optional result limit

        Returns:
            List of entity data dictionaries
        """
        try:
            query, params = self.query_builder.build_find_entity_query(
                entity_type, filters, limit
            )

            async with self.driver.session() as session:
                result = await session.run(query, params)
                entities = []

                async for record in result:
                    entity_data = dict(record["e"])
                    entities.append(entity_data)

                logger.info(f"✅ Found {len(entities)} entities of type {entity_type}")
                return entities

        except Exception as e:
            logger.error(f"❌ Failed to find entities of type {entity_type}: {e}")
            return []

    async def search_entities_fulltext(
        self, entity_types: List[str], query_text: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Perform full-text search across entity types.

        Args:
            entity_types: List of entity types to search
            query_text: Search query text
            limit: Maximum number of results

        Returns:
            List of search results with scores
        """
        try:
            query, params = self.query_builder.build_fulltext_search_query(
                entity_types, query_text, limit
            )

            async with self.driver.session() as session:
                result = await session.run(query, params)
                search_results = []

                async for record in result:
                    entity_data = dict(record["node"])
                    entity_data["_score"] = record["score"]
                    entity_data["_type"] = record["entity_type"]
                    search_results.append(entity_data)

                logger.info(
                    f"✅ Full-text search returned {len(search_results)} results"
                )
                return search_results

        except Exception as e:
            logger.error(f"❌ Full-text search failed: {e}")
            return []

    async def get_entity_relationships(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for an entity.

        Args:
            entity_id: ID of the entity
            relationship_types: Optional list of relationship types to filter
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of relationship data with connected entities
        """
        try:
            query, params = self.query_builder.build_get_entity_relationships_query(
                entity_id, relationship_types, direction
            )

            async with self.driver.session() as session:
                result = await session.run(query, params)
                relationships = []

                async for record in result:
                    rel_data = {
                        "relationship": dict(record["r"]),
                        "other_entity": dict(record["other"]),
                        "relationship_type": record["r"].type,
                    }
                    relationships.append(rel_data)

                logger.info(
                    f"✅ Found {len(relationships)} relationships for entity {entity_id}"
                )
                return relationships

        except Exception as e:
            logger.error(f"❌ Failed to get relationships for {entity_id}: {e}")
            return []

    async def update_entity(
        self, entity_id: str, updates: Dict[str, Any], validate: bool = True
    ) -> bool:
        """
        Update entity properties.

        Args:
            entity_id: ID of the entity to update
            updates: Dictionary of property updates
            validate: Whether to validate updates against schema

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            # Validate updates if requested
            if validate:
                entity_type = await self.get_entity_type(entity_id)
                if entity_type:
                    validation_result = self.schema.validate_entity_data(
                        entity_type, updates
                    )
                    if not validation_result.is_valid:
                        logger.error(
                            f"Update validation failed: {validation_result.errors}"
                        )
                        return False

            query, params = self.query_builder.build_update_entity_query(
                entity_id, updates
            )

            async with self.driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()

                if record:
                    logger.info(f"✅ Updated entity {entity_id}")
                    return True

                return False

        except Exception as e:
            logger.error(f"❌ Failed to update entity {entity_id}: {e}")
            return False

    async def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity and all its relationships.

        Args:
            entity_id: ID of the entity to delete

        Returns:
            True if deletion succeeded, False otherwise
        """
        try:
            query, params = self.query_builder.build_delete_entity_query(entity_id)

            async with self.driver.session() as session:
                await session.run(query, params)
                logger.info(f"✅ Deleted entity {entity_id}")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to delete entity {entity_id}: {e}")
            return False

    async def get_global_glossary(self) -> str:
        """
        Build a global glossary from all definitions in the graph.

        Returns:
            Formatted glossary string
        """
        try:
            query, params = self.query_builder.build_get_glossary_query()

            async with self.driver.session() as session:
                result = await session.run(query, params)
                glossary_items = []

                async for record in result:
                    term = record["term"]
                    definition = record["definition"]
                    item_type = record["type"]

                    if term and definition:
                        glossary_items.append(
                            f"- {item_type}: {term}\n  Definition: {definition}"
                        )

                return "\n".join(glossary_items)

        except Exception as e:
            logger.error(f"❌ Failed to build global glossary: {e}")
            return ""

    async def get_source_glossary(self, source_id: str) -> str:
        """
        Build a glossary for a specific source document.

        Args:
            source_id: ID of the source document

        Returns:
            Formatted glossary string for the source
        """
        try:
            query, params = self.query_builder.build_get_glossary_query(source_id)

            async with self.driver.session() as session:
                result = await session.run(query, params)
                glossary_items = []

                async for record in result:
                    term = record["term"]
                    definition = record["definition"]
                    item_type = record["type"]

                    if term and definition:
                        glossary_items.append(
                            f"- {item_type}: {term}\n  Definition: {definition}"
                        )

                return "\n".join(glossary_items)

        except Exception as e:
            logger.error(f"❌ Failed to build source glossary for {source_id}: {e}")
            return ""

    async def count_entities(self, entity_type: Optional[str] = None) -> int:
        """
        Count entities in the graph, optionally by type.

        Args:
            entity_type: Optional entity type to count

        Returns:
            Number of entities
        """
        try:
            query, params = self.query_builder.build_count_entities_query(entity_type)

            async with self.driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()

                if record:
                    return record["count"]

                return 0

        except Exception as e:
            logger.error(f"❌ Failed to count entities: {e}")
            return 0

    def _generate_entity_id(self, entity_type: str) -> str:
        """
        Generate a unique ID for an entity.

        Args:
            entity_type: Type of entity

        Returns:
            Generated unique ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = str(uuid.uuid4())[:8]
        return f"{entity_type.lower()}_{timestamp}_{random_part}"

    async def add_source(self, source_data: Dict[str, Any]) -> str:
        """
        Add a source document to the graph.

        Args:
            source_data: Source document data

        Returns:
            ID of the created source
        """
        return await self.add_entity("Source", source_data)
