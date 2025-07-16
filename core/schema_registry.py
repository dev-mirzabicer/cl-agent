# core/schema_registry.py
"""
Centralized schema registry for managing graph schema configuration.
Handles loading, validation, and access to schema definitions.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

from models.schema_models import (
    GraphSchema,
    EntitySchema,
    RelationshipSchema,
    ValidationResult,
    PropertyDefinition,
    ExtractionStrategy,
    EntityData,
    RelationshipData,
)

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """
    Central registry for managing graph schema configuration.
    Provides validation, access, and management of schema definitions.
    """

    def __init__(self, config_path: str = "config/graph_schema.yaml"):
        """
        Initialize the schema registry.

        Args:
            config_path: Path to the schema configuration file
        """
        self.config_path = Path(config_path)
        self.schema: Optional[GraphSchema] = None
        self._entity_schemas: Dict[str, EntitySchema] = {}
        self._relationship_schemas: Dict[str, RelationshipSchema] = {}

        # Load the schema configuration
        self.load_schema()

    def load_schema(self) -> None:
        """Load and validate the schema configuration from file."""
        try:
            logger.info(f"Loading schema configuration from {self.config_path}")

            if not self.config_path.exists():
                raise FileNotFoundError(
                    f"Schema configuration file not found: {self.config_path}"
                )

            # Load YAML configuration
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Inject entity and relationship names from keys into the dictionary values
            if "entities" in config_data and config_data["entities"]:
                for name, entity_data in config_data["entities"].items():
                    if "name" not in entity_data:
                        entity_data["name"] = name

            if "relationships" in config_data and config_data["relationships"]:
                for name, rel_data in config_data["relationships"].items():
                    if "name" not in rel_data:
                        rel_data["name"] = name

            # Validate and create schema object
            self.schema = GraphSchema(**config_data)

            # Cache entity and relationship schemas for quick access
            self._entity_schemas = self.schema.entities
            self._relationship_schemas = self.schema.relationships

            logger.info(
                f"✅ Schema loaded successfully: {len(self._entity_schemas)} entities, {len(self._relationship_schemas)} relationships"
            )

        except Exception as e:
            logger.error(f"❌ Failed to load schema configuration: {e}")
            raise

    def reload_schema(self) -> None:
        """Reload the schema configuration from file."""
        logger.info("🔄 Reloading schema configuration...")
        self.load_schema()

    def get_entity_schema(self, entity_type: str) -> Optional[EntitySchema]:
        """
        Get the schema definition for an entity type.

        Args:
            entity_type: Name of the entity type

        Returns:
            EntitySchema object or None if not found
        """
        return self._entity_schemas.get(entity_type)

    def get_relationship_schema(
        self, relationship_type: str
    ) -> Optional[RelationshipSchema]:
        """
        Get the schema definition for a relationship type.

        Args:
            relationship_type: Name of the relationship type

        Returns:
            RelationshipSchema object or None if not found
        """
        return self._relationship_schemas.get(relationship_type)

    def get_all_entity_types(self) -> List[str]:
        """Get all defined entity type names."""
        return list(self._entity_schemas.keys())

    def get_all_relationship_types(self) -> List[str]:
        """Get all defined relationship type names."""
        return list(self._relationship_schemas.keys())

    def get_extraction_strategy(self) -> ExtractionStrategy:
        """Get the extraction strategy configuration."""
        if not self.schema:
            raise ValueError("Schema not loaded")
        return self.schema.extraction_strategy

    def get_neo4j_setup_commands(self) -> Dict[str, List[str]]:
        """Get Neo4j setup commands for constraints and indexes."""
        if not self.schema:
            raise ValueError("Schema not loaded")

        return {
            "constraints": self.schema.neo4j_setup.constraints,
            "indexes": self.schema.neo4j_setup.indexes,
        }

    def validate_entity_data(
        self, entity_type: str, data: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate entity data against its schema definition.

        Args:
            entity_type: Type of entity to validate
            data: Entity data to validate

        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult(is_valid=True)

        # Get entity schema
        entity_schema = self.get_entity_schema(entity_type)
        if not entity_schema:
            result.add_error(f"Unknown entity type: {entity_type}")
            return result

        # Check required properties
        for prop_name, prop_def in entity_schema.properties.items():
            if prop_def.required and prop_name not in data:
                result.add_error(
                    f"Required property '{prop_name}' missing for {entity_type}"
                )

        # Validate each provided property
        for prop_name, prop_value in data.items():
            if prop_name not in entity_schema.properties:
                result.add_warning(f"Unknown property '{prop_name}' for {entity_type}")
                continue

            prop_def = entity_schema.properties[prop_name]
            validation_errors = self._validate_property_value(
                prop_name, prop_value, prop_def
            )
            result.errors.extend(validation_errors)
            if validation_errors:
                result.is_valid = False

        return result

    def validate_relationship_data(
        self,
        relationship_type: str,
        source_entity_type: str,
        target_entity_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate relationship data against its schema definition.

        Args:
            relationship_type: Type of relationship
            source_entity_type: Type of source entity
            target_entity_type: Type of target entity
            data: Additional relationship data

        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult(is_valid=True)

        # Get relationship schema
        rel_schema = self.get_relationship_schema(relationship_type)
        if not rel_schema:
            result.add_error(f"Unknown relationship type: {relationship_type}")
            return result

        # Validate source entity type
        if source_entity_type not in rel_schema.source_entities:
            result.add_error(
                f"Entity type '{source_entity_type}' cannot be source for relationship '{relationship_type}'"
            )

        # Validate target entity type
        if target_entity_type not in rel_schema.target_entities:
            result.add_error(
                f"Entity type '{target_entity_type}' cannot be target for relationship '{relationship_type}'"
            )

        # Validate relationship properties if provided
        if data:
            for prop_name, prop_value in data.items():
                if prop_name not in rel_schema.properties:
                    result.add_warning(
                        f"Unknown property '{prop_name}' for relationship '{relationship_type}'"
                    )
                    continue

                prop_def = rel_schema.properties[prop_name]
                validation_errors = self._validate_property_value(
                    prop_name, prop_value, prop_def
                )
                result.errors.extend(validation_errors)
                if validation_errors:
                    result.is_valid = False

        return result

    def _validate_property_value(
        self, prop_name: str, value: Any, prop_def: PropertyDefinition
    ) -> List[str]:
        """
        Validate a single property value against its definition.

        Args:
            prop_name: Name of the property
            value: Value to validate
            prop_def: Property definition

        Returns:
            List of validation error messages
        """
        errors = []

        # Type validation
        if prop_def.type.value == "string":
            if not isinstance(value, str):
                errors.append(f"Property '{prop_name}' must be a string")
        elif prop_def.type.value == "text":
            if not isinstance(value, str):
                errors.append(f"Property '{prop_name}' must be text (string)")
        elif prop_def.type.value == "integer":
            if not isinstance(value, int):
                errors.append(f"Property '{prop_name}' must be an integer")
        elif prop_def.type.value == "float":
            if not isinstance(value, (int, float)):
                errors.append(f"Property '{prop_name}' must be a number")
        elif prop_def.type.value == "boolean":
            if not isinstance(value, bool):
                errors.append(f"Property '{prop_name}' must be a boolean")
        elif prop_def.type.value == "list":
            if not isinstance(value, list):
                errors.append(f"Property '{prop_name}' must be a list")

        # Validation rules
        if prop_def.validation_rules and not errors:  # Only validate if type is correct
            rules = prop_def.validation_rules

            # Numeric range validation
            if rules.min is not None and isinstance(value, (int, float)):
                if value < rules.min:
                    errors.append(f"Property '{prop_name}' must be >= {rules.min}")

            if rules.max is not None and isinstance(value, (int, float)):
                if value > rules.max:
                    errors.append(f"Property '{prop_name}' must be <= {rules.max}")

            # String length validation
            if rules.min_length is not None and isinstance(value, str):
                if len(value) < rules.min_length:
                    errors.append(
                        f"Property '{prop_name}' must be at least {rules.min_length} characters"
                    )

            if rules.max_length is not None and isinstance(value, str):
                if len(value) > rules.max_length:
                    errors.append(
                        f"Property '{prop_name}' must be at most {rules.max_length} characters"
                    )

            # Pattern validation
            if rules.pattern and isinstance(value, str):
                import re

                if not re.match(rules.pattern, value):
                    errors.append(
                        f"Property '{prop_name}' does not match required pattern"
                    )

            # Enum validation
            if rules.enum and value not in rules.enum:
                errors.append(f"Property '{prop_name}' must be one of: {rules.enum}")

        return errors

    def get_entity_llm_instructions(self, entity_type: str) -> str:
        """Get LLM instructions for an entity type."""
        entity_schema = self.get_entity_schema(entity_type)
        if not entity_schema:
            return f"Unknown entity type: {entity_type}"
        return entity_schema.llm_instructions

    def get_relationship_llm_instructions(self, relationship_type: str) -> str:
        """Get LLM instructions for a relationship type."""
        rel_schema = self.get_relationship_schema(relationship_type)
        if not rel_schema:
            return f"Unknown relationship type: {relationship_type}"
        return rel_schema.llm_instructions

    def get_entity_properties_description(self, entity_type: str) -> str:
        """Get a description of all properties for an entity type."""
        entity_schema = self.get_entity_schema(entity_type)
        if not entity_schema:
            return f"Unknown entity type: {entity_type}"

        descriptions = []
        for prop_name, prop_def in entity_schema.properties.items():
            required_text = " (required)" if prop_def.required else " (optional)"
            desc = f"- {prop_name}: {prop_def.type.value}{required_text}"
            if prop_def.description:
                desc += f" - {prop_def.description}"
            descriptions.append(desc)

        return "\n".join(descriptions)

    def get_relationships_for_entity(self, entity_type: str) -> Dict[str, List[str]]:
        """
        Get all relationships that an entity type can participate in.

        Args:
            entity_type: Name of the entity type

        Returns:
            Dict with 'outgoing' and 'incoming' relationship lists
        """
        entity_schema = self.get_entity_schema(entity_type)
        if not entity_schema:
            return {"outgoing": [], "incoming": []}

        return {
            "outgoing": entity_schema.relationships.get("outgoing", []),
            "incoming": entity_schema.relationships.get("incoming", []),
        }

    def export_schema_summary(self) -> Dict[str, Any]:
        """Export a summary of the current schema for debugging/inspection."""
        if not self.schema:
            return {"error": "Schema not loaded"}

        return {
            "entities": {
                name: {
                    "description": schema.description,
                    "properties": len(schema.properties),
                    "outgoing_relationships": schema.relationships.get("outgoing", []),
                    "incoming_relationships": schema.relationships.get("incoming", []),
                }
                for name, schema in self._entity_schemas.items()
            },
            "relationships": {
                name: {
                    "description": schema.description,
                    "source_entities": schema.source_entities,
                    "target_entities": schema.target_entities,
                }
                for name, schema in self._relationship_schemas.items()
            },
            "extraction_passes": len(self.schema.extraction_strategy.passes),
        }
