# prompts/dynamic_prompts.py
"""
Dynamic prompt generation engine that creates LLM prompts based on schema configuration.
Integrates with graphiti retrieval to provide relevant context from the existing graph.
"""

from typing import List, Dict, Any, Optional
import logging
from jinja2 import Environment, BaseLoader, Template

from core.schema_registry import SchemaRegistry
from core.dynamic_graph_manager import DynamicGraphManager

logger = logging.getLogger(__name__)


class StringTemplateLoader(BaseLoader):
    """Simple Jinja2 loader for string templates."""

    def __init__(self, templates: Dict[str, str]):
        self.templates = templates

    def get_source(self, environment, template):
        if template in self.templates:
            source = self.templates[template]
            return source, None, lambda: True
        raise jinja2.TemplateNotFound(template)


class DynamicPromptEngine:
    """
    Generates extraction prompts dynamically based on schema configuration.
    Integrates with graphiti to retrieve relevant context from the graph.
    """

    def __init__(
        self,
        schema_registry: SchemaRegistry,
        graph_manager: DynamicGraphManager,
        graphiti_client=None,
    ):
        """
        Initialize the prompt engine.

        Args:
            schema_registry: Registry containing schema definitions
            graph_manager: Graph manager for context retrieval
            graphiti_client: Graphiti client for advanced retrieval
        """
        self.schema = schema_registry
        self.graph_manager = graph_manager
        self.graphiti = graphiti_client

        # Setup Jinja2 environment with templates
        self.templates = self._load_templates()
        self.jinja_env = Environment(loader=StringTemplateLoader(self.templates))

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates."""
        return {
            "entity_extraction": self._get_entity_extraction_template(),
            "relationship_instructions": self._get_relationship_instructions_template(),
            "context_section": self._get_context_section_template(),
        }

    def _get_entity_extraction_template(self) -> str:
        """Base template for entity extraction."""
        return """
You are an expert in mathematical logic and combinatory logic. Your task is to analyze the provided text and extract specific types of mathematical entities with high precision.

**CRITICAL INSTRUCTIONS:**
- Only extract content that clearly fits the entity definitions provided
- Be conservative: when in doubt, don't extract
- Ensure all extracted entities are self-contained and meaningful
- Use the provided contextual information to understand terminology correctly

{% if context_section %}
**CONTEXTUAL INFORMATION:**
{{ context_section }}
{% endif %}

**TARGET ENTITIES:**
{% for entity in entities %}
**{{ entity.name }}:**
{{ entity.llm_instructions }}

Required Properties:
{{ entity.properties_description }}

Allowed Relationships:
{{ entity.relationships_description }}

{% endfor %}

**RELATIONSHIP TYPES:**
{% for relationship in relationships %}
**{{ relationship.name }}:**
{{ relationship.llm_instructions }}
Valid connections: {{ relationship.source_entities|join(', ') }} → {{ relationship.target_entities|join(', ') }}

{% endfor %}

**OUTPUT FORMAT:**
Provide a JSON object with a single key "extracted_entities" containing a list of all entities you found.

Each entity must include:
- type: One of [{{ target_entity_types|join(', ') }}]
- id: A unique identifier (e.g., "fact_church_rosser_theorem")
- All required properties for that entity type
- Relationship references using the target entity IDs

**EXAMPLE:**
```json
{
  "extracted_entities": [
    {
      "type": "Definition",
      "id": "def_combinator",
      "term": "Combinator",
      "definition": "A lambda term with no free variables.",
      "informal_explanation": "Functions that are complete in themselves."
    },
    {
      "type": "Fact", 
      "id": "fact_church_rosser",
      "content": "The lambda calculus satisfies the Church-Rosser property.",
      "explanation": "This ensures confluence of reduction.",
      "statement_type": "theorem",
      "uses_definitions": ["def_combinator"]
    }
  ]
}
```

**TEXT TO ANALYZE:**
{{ text_content }}
"""

    def _get_relationship_instructions_template(self) -> str:
        """Template for relationship instructions."""
        return """
When creating relationships between entities:
{% for rel in relationships %}
- **{{ rel.name }}**: {{ rel.llm_instructions }}
{% endfor %}
"""

    def _get_context_section_template(self) -> str:
        """Template for context section."""
        return """
The following definitions and concepts have already been established in this document or related sources:

{% for item in context_items %}
**{{ item.type }}: {{ item.term or item.content[:100] }}**
{{ item.definition or item.explanation }}

{% endfor %}

Use this context to understand the text and ensure consistency in your extractions.
"""

    async def generate_extraction_prompt(
        self,
        target_entities: List[str],
        text_content: str,
        context_query: Optional[str] = None,
        max_context_items: int = 50,
    ) -> str:
        """
        Generate a dynamic extraction prompt for the specified entity types.

        Args:
            target_entities: List of entity types to extract
            text_content: Text content to analyze
            context_query: Optional query to retrieve relevant context
            max_context_items: Maximum number of context items to include

        Returns:
            Complete extraction prompt string
        """
        logger.info(f"Generating extraction prompt for entities: {target_entities}")

        # Build entity descriptions
        entities = []
        for entity_type in target_entities:
            entity_schema = self.schema.get_entity_schema(entity_type)
            if entity_schema:
                entities.append(
                    {
                        "name": entity_type,
                        "llm_instructions": entity_schema.llm_instructions,
                        "properties_description": self._build_properties_description(
                            entity_schema
                        ),
                        "relationships_description": self._build_relationships_description(
                            entity_type
                        ),
                    }
                )

        # Build relationship descriptions for target entities
        relationships = self._get_relevant_relationships(target_entities)

        # Retrieve context if needed
        context_section = ""
        if context_query and self.graphiti:
            context_section = await self._retrieve_context_section(
                context_query, max_context_items
            )

        # Render the template
        template = self.jinja_env.get_template("entity_extraction")
        prompt = template.render(
            entities=entities,
            relationships=relationships,
            target_entity_types=target_entities,
            context_section=context_section,
            text_content=text_content,
        )

        logger.debug(f"Generated prompt length: {len(prompt)} characters")
        return prompt

    async def _retrieve_context_section(self, query: str, max_items: int) -> str:
        """
        Retrieve relevant context from the graph using graphiti.

        Args:
            query: Search query for context retrieval
            max_items: Maximum number of context items

        Returns:
            Formatted context section
        """
        try:
            logger.info(f"Retrieving context for query: {query}")

            if not self.graphiti:
                logger.warning("Graphiti client not available for context retrieval")
                return ""

            # Use graphiti's search capabilities
            from graphiti_core.search.search_config import (
                SearchConfig,
                NodeSearchConfig,
            )
            from graphiti_core.search.search import NodeSearchMethod, NodeReranker

            search_config = SearchConfig(
                node_config=NodeSearchConfig(
                    search_methods=[
                        NodeSearchMethod.cosine_similarity,
                        NodeSearchMethod.bm25,
                    ],
                    reranker=NodeReranker.cross_encoder,
                ),
                limit=max_items,
            )

            search_results = await self.graphiti.search_(
                query=query,
                config=search_config,
                group_ids=["combinatory-logic-books", "combinatory-logic-papers"],
            )

            # Format context items
            context_items = []
            for node in search_results.nodes:
                node_data = node.dict()
                context_item = {
                    "type": node_data.get("uuid", "").split("_")[0]
                    if "_" in node_data.get("uuid", "")
                    else "Unknown",
                    "term": node_data.get("term") or node_data.get("label"),
                    "content": node_data.get("content", ""),
                    "definition": node_data.get("definition"),
                    "explanation": node_data.get("explanation"),
                }
                context_items.append(context_item)

            if context_items:
                context_template = self.jinja_env.get_template("context_section")
                context_section = context_template.render(context_items=context_items)
                logger.info(f"Retrieved {len(context_items)} context items")
                return context_section
            else:
                logger.info("No context items found")
                return ""

        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return ""

    def _build_properties_description(self, entity_schema) -> str:
        """Build a description of entity properties."""
        descriptions = []
        for prop_name, prop_def in entity_schema.properties.items():
            required_text = " (required)" if prop_def.required else " (optional)"
            desc = f"- {prop_name}: {prop_def.type.value}{required_text}"
            if prop_def.description:
                desc += f" - {prop_def.description}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def _build_relationships_description(self, entity_type: str) -> str:
        """Build a description of allowed relationships for an entity type."""
        relationships = self.schema.get_relationships_for_entity(entity_type)

        descriptions = []
        if relationships.get("outgoing"):
            descriptions.append(
                f"Can reference: {', '.join(relationships['outgoing'])}"
            )
        if relationships.get("incoming"):
            descriptions.append(
                f"Can be referenced by: {', '.join(relationships['incoming'])}"
            )

        return (
            "; ".join(descriptions)
            if descriptions
            else "No specific relationship constraints"
        )

    def _get_relevant_relationships(
        self, target_entities: List[str]
    ) -> List[Dict[str, Any]]:
        """Get relationship schemas relevant to the target entities."""
        relevant_relationships = []

        for rel_name, rel_schema in self.schema._relationship_schemas.items():
            # Check if any target entity can participate in this relationship
            entities_can_use = any(
                entity in rel_schema.source_entities
                or entity in rel_schema.target_entities
                for entity in target_entities
            )

            if entities_can_use:
                relevant_relationships.append(
                    {
                        "name": rel_name,
                        "llm_instructions": rel_schema.llm_instructions,
                        "source_entities": rel_schema.source_entities,
                        "target_entities": rel_schema.target_entities,
                    }
                )

        return relevant_relationships

    async def generate_pass_specific_prompt(
        self, pass_config, text_content: str, existing_glossary: str = ""
    ) -> str:
        """
        Generate a prompt for a specific extraction pass.

        Args:
            pass_config: Extraction pass configuration
            text_content: Text to analyze
            existing_glossary: Previously extracted glossary items

        Returns:
            Extraction prompt for the pass
        """
        logger.info(f"Generating prompt for pass: {pass_config.name}")

        # Determine context query based on pass configuration
        context_query = None
        if (
            pass_config.requires_context
            and pass_config.context_retrieval
            and pass_config.context_retrieval.enabled
        ):
            # Create a query from the text content for context retrieval
            context_query = self._create_context_query(
                text_content, pass_config.context_entities
            )

        # Generate the prompt
        prompt = await self.generate_extraction_prompt(
            target_entities=pass_config.target_entities,
            text_content=text_content,
            context_query=context_query,
            max_context_items=pass_config.context_retrieval.max_context_items
            if pass_config.context_retrieval
            else 50,
        )

        # Add existing glossary if provided
        if existing_glossary:
            glossary_section = f"\n\n**EXISTING GLOSSARY:**\n{existing_glossary}\n"
            prompt = prompt.replace(
                "**TEXT TO ANALYZE:**", glossary_section + "**TEXT TO ANALYZE:**"
            )

        return prompt

    def _create_context_query(
        self, text_content: str, context_entities: List[str]
    ) -> str:
        """
        Create a context query from text content and target context entities.

        Args:
            text_content: Text being processed
            context_entities: Entity types to use as context

        Returns:
            Query string for context retrieval
        """
        # Extract key terms from the text for context search
        # This is a simplified approach - could be enhanced with NLP
        words = text_content.lower().split()

        # Look for mathematical/logical terms
        math_terms = []
        keywords = [
            "combinator",
            "lambda",
            "calculus",
            "reduction",
            "normal",
            "form",
            "theorem",
            "proof",
            "lemma",
            "definition",
            "axiom",
            "rule",
            "function",
            "application",
            "abstraction",
            "variable",
            "term",
        ]

        for word in words:
            cleaned_word = word.strip(".,;:!?()[]{}\"'")
            if cleaned_word in keywords or len(cleaned_word) > 5:
                if cleaned_word not in math_terms:
                    math_terms.append(cleaned_word)

        # Create a query from the most relevant terms
        query_terms = math_terms[:10]  # Limit to top 10 terms
        return " ".join(query_terms)

    def get_entity_schema_summary(self) -> str:
        """Get a summary of all entity schemas for debugging."""
        summary = []
        for entity_type in self.schema.get_all_entity_types():
            entity_schema = self.schema.get_entity_schema(entity_type)
            summary.append(f"**{entity_type}**: {entity_schema.description}")

        return "\n".join(summary)

    def get_relationship_schema_summary(self) -> str:
        """Get a summary of all relationship schemas for debugging."""
        summary = []
        for rel_type in self.schema.get_all_relationship_types():
            rel_schema = self.schema.get_relationship_schema(rel_type)
            summary.append(f"**{rel_type}**: {rel_schema.description}")

        return "\n".join(summary)
