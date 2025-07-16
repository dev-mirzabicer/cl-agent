# models/schema_models.py
"""
Type-safe schema models for the dynamic graph management system.
Provides Pydantic models for validation and type checking.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import re


class PropertyType(str, Enum):
    """Supported property types in the schema."""

    STRING = "string"
    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    LIST = "list"


class ValidationRules(BaseModel):
    """Validation rules for properties."""

    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    enum: Optional[List[str]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class PropertyDefinition(BaseModel):
    """Definition of a property within an entity schema."""

    type: PropertyType
    required: bool = False
    unique: bool = False
    indexed: bool = False
    description: Optional[str] = None
    default: Optional[Union[str, int, float, bool, List]] = None
    validation_rules: Optional[ValidationRules] = None

    @field_validator("validation_rules")
    def validate_rules_for_type(
        cls, v: Optional[ValidationRules], info: ValidationInfo
    ):
        """Ensure validation rules are appropriate for the property type."""
        if v is None:
            return v

        prop_type = info.data.get("type")
        if prop_type in [PropertyType.INTEGER, PropertyType.FLOAT]:
            # Numeric types can have min/max
            pass
        elif prop_type in [PropertyType.STRING, PropertyType.TEXT]:
            # String types can have length constraints and patterns
            if v.min is not None or v.max is not None:
                raise ValueError(
                    "String types should use min_length/max_length, not min/max"
                )

        return v


class ExtractionRules(BaseModel):
    """Rules for extracting entities from text."""

    indicators: List[str] = Field(
        default_factory=list, description="Keywords that indicate this entity type"
    )
    context_required: bool = False
    requires_proof_context: bool = False
    requires_fact_reference: bool = False


class RelationshipProperties(BaseModel):
    """Properties that can be attached to relationships."""

    properties: Dict[str, PropertyDefinition] = Field(default_factory=dict)


class RelationshipSchema(BaseModel):
    """Schema definition for a relationship type."""

    name: str = Field(..., description="Name of the relationship")
    description: str = Field(..., description="Human-readable description")
    llm_instructions: str = Field(
        ..., description="Instructions for LLM on when to use this relationship"
    )
    source_entities: List[str] = Field(..., description="Valid source entity types")
    target_entities: List[str] = Field(..., description="Valid target entity types")
    properties: Dict[str, PropertyDefinition] = Field(default_factory=dict)

    @field_validator("name")
    def validate_name(cls, v):
        """Ensure relationship name is valid."""
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", v):
            raise ValueError("Relationship name must be uppercase with underscores")
        return v


class EntitySchema(BaseModel):
    """Schema definition for an entity type."""

    name: str = Field(..., description="Name of the entity type")
    description: str = Field(..., description="Human-readable description")
    llm_instructions: str = Field(
        ..., description="Detailed instructions for LLM extraction"
    )
    properties: Dict[str, PropertyDefinition] = Field(
        ..., description="Property definitions"
    )
    relationships: Dict[str, List[str]] = Field(
        default_factory=dict, description="Allowed relationships"
    )
    extraction_rules: Optional[ExtractionRules] = None

    @field_validator("name")
    def validate_name(cls, v):
        """Ensure entity name follows conventions."""
        if not v[0].isupper() or not v.replace("_", "").isalnum():
            raise ValueError(
                "Entity name must start with uppercase letter and contain only letters, numbers, underscores"
            )
        return v

    @field_validator("relationships")
    def validate_relationships(cls, v):
        """Ensure relationships dict has correct structure."""
        allowed_keys = {"outgoing", "incoming"}
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Relationship key must be one of: {allowed_keys}")
        return v

    @field_validator("properties")
    def validate_required_id(cls, v):
        """Ensure every entity has an id property."""
        if "id" not in v:
            raise ValueError('Every entity must have an "id" property')

        id_prop = v["id"]
        if not id_prop.required or not id_prop.unique:
            raise ValueError('Entity "id" property must be required and unique')

        return v


class ContextRetrievalConfig(BaseModel):
    """Configuration for context retrieval during extraction."""

    enabled: bool = False
    max_context_items: int = 50
    search_methods: List[str] = Field(
        default_factory=lambda: ["bm25", "cosine_similarity"]
    )


class ExtractionPass(BaseModel):
    """Configuration for a single extraction pass."""

    name: str = Field(..., description="Name of the extraction pass")
    description: str = Field(..., description="Description of what this pass does")
    target_entities: List[str] = Field(
        ..., description="Entity types to extract in this pass"
    )
    context_entities: List[str] = Field(
        default_factory=list, description="Entity types to use as context"
    )
    requires_context: bool = False
    context_retrieval: Optional[ContextRetrievalConfig] = None


class ExtractionStrategy(BaseModel):
    """Complete extraction strategy configuration."""

    passes: List[ExtractionPass] = Field(
        ..., description="Ordered list of extraction passes"
    )

    @field_validator("passes")
    def validate_passes_not_empty(cls, v):
        """Ensure at least one extraction pass is defined."""
        if not v:
            raise ValueError("At least one extraction pass must be defined")
        return v


class Neo4jSetup(BaseModel):
    """Neo4j database setup configuration."""

    constraints: List[str] = Field(
        default_factory=list, description="Cypher queries to create constraints"
    )
    indexes: List[str] = Field(
        default_factory=list, description="Cypher queries to create indexes"
    )


class GraphSchema(BaseModel):
    """Complete graph schema configuration."""

    entities: Dict[str, EntitySchema] = Field(
        ..., description="Entity type definitions"
    )
    relationships: Dict[str, RelationshipSchema] = Field(
        ..., description="Relationship type definitions"
    )
    extraction_strategy: ExtractionStrategy = Field(
        ..., description="Multi-pass extraction configuration"
    )
    neo4j_setup: Neo4jSetup = Field(
        default_factory=Neo4jSetup, description="Database setup configuration"
    )

    @model_validator(mode="after")
    def validate_relationship_entity_references(self):
        """Ensure relationship schemas reference valid entity types."""
        entities = self.entities
        relationships = self.relationships

        entity_names = set(entities.keys())

        for rel_name, rel_schema in relationships.items():
            # Check source entities
            for source_entity in rel_schema.source_entities:
                if source_entity not in entity_names:
                    raise ValueError(
                        f"Relationship {rel_name} references unknown source entity: {source_entity}"
                    )

            # Check target entities
            for target_entity in rel_schema.target_entities:
                if target_entity not in entity_names:
                    raise ValueError(
                        f"Relationship {rel_name} references unknown target entity: {target_entity}"
                    )

        return self

    @model_validator(mode="after")
    def validate_extraction_strategy_entities(self):
        """Ensure extraction strategy references valid entity types."""
        entities = self.entities
        extraction_strategy = self.extraction_strategy

        if not extraction_strategy:
            return self

        entity_names = set(entities.keys())

        for pass_config in extraction_strategy.passes:
            # Check target entities
            for entity_type in pass_config.target_entities:
                if entity_type not in entity_names:
                    raise ValueError(
                        f'Extraction pass "{pass_config.name}" references unknown target entity: {entity_type}'
                    )

            # Check context entities
            for entity_type in pass_config.context_entities:
                if entity_type not in entity_names:
                    raise ValueError(
                        f'Extraction pass "{pass_config.name}" references unknown context entity: {entity_type}'
                    )

        return self


class ValidationResult(BaseModel):
    """Result of schema or data validation."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def add_error(self, error: str):
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning to the validation result."""
        self.warnings.append(warning)


class EntityData(BaseModel):
    """Base model for entity data validation."""

    id: str = Field(..., description="Unique identifier for the entity")

    class Config:
        extra = "allow"  # Allow additional fields based on schema


class RelationshipData(BaseModel):
    """Base model for relationship data validation."""

    source_id: str = Field(..., description="ID of the source entity")
    target_id: str = Field(..., description="ID of the target entity")
    relationship_type: str = Field(..., description="Type of relationship")

    class Config:
        extra = "allow"  # Allow additional fields based on schema
