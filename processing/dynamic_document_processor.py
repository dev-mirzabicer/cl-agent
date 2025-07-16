# processing/dynamic_document_processor.py
"""
Schema-driven document processor that performs multi-pass extraction
using dynamic prompts and graph-aware context retrieval.
"""

import asyncio
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import demjson3

import pymupdf
import pymupdf4llm
from pylatexenc.latex2text import LatexNodes2Text

from core.schema_registry import SchemaRegistry
from core.dynamic_graph_manager import DynamicGraphManager
from prompts.dynamic_prompts import DynamicPromptEngine
from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata for processed documents."""

    title: str
    authors: List[str]
    publication_year: int
    document_type: str  # "book", "paper", "chapter"
    source_path: str
    page_range: Optional[tuple] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None


@dataclass
class DocumentSection:
    """Represents a section of a document."""

    title: str
    content: str
    level: int
    page_start: int
    page_end: int
    parent_titles: List[str]


class DynamicDocumentProcessor:
    """
    Processes documents using schema-driven extraction with multiple passes.
    Integrates with graphiti for context-aware extraction.
    """

    def __init__(self, system, schema_registry: SchemaRegistry):
        """
        Initialize the document processor.

        Args:
            system: System instance with graphiti client
            schema_registry: Schema registry for validation and configuration
        """
        self.system = system
        self.schema = schema_registry
        self.graph_manager = DynamicGraphManager(
            system.graphiti.driver, schema_registry
        )
        self.prompt_engine = DynamicPromptEngine(
            schema_registry, self.graph_manager, system.graphiti
        )

    async def process_document(self, file_path: str, metadata: DocumentMetadata) -> int:
        """
        Process a single document using multi-pass extraction.

        Args:
            file_path: Path to the document file
            metadata: Document metadata

        Returns:
            Total number of extracted items
        """
        logger.info(f"📄 Processing document: {metadata.title}")

        # Extract text and structure
        text_content, sections = await self._extract_document_content(file_path)

        # Add source to graph
        source_id = await self._add_source_to_graph(metadata)

        # Execute multi-pass extraction strategy
        extraction_strategy = self.schema.get_extraction_strategy()
        total_items = 0

        for pass_idx, pass_config in enumerate(extraction_strategy.passes):
            logger.info(f"\n--- Starting Pass {pass_idx + 1}: {pass_config.name} ---")

            items_extracted = await self._execute_extraction_pass(
                pass_config, sections, source_id, metadata
            )

            total_items += items_extracted
            logger.info(
                f"--- Pass {pass_idx + 1} Complete: {items_extracted} items extracted ---"
            )

        logger.info(
            f"✅ Document processing complete: {total_items} total items extracted"
        )
        return total_items

    async def _extract_document_content(
        self, file_path: str
    ) -> Tuple[str, List[DocumentSection]]:
        """
        Extract text content and structure from a document.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (full_text, document_sections)
        """
        logger.info(f"📖 Extracting content from: {file_path}")

        if file_path.endswith(".pdf"):
            return self._process_pdf(file_path)
        elif file_path.endswith(".tex"):
            return self._process_latex(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _process_pdf(self, file_path: str) -> Tuple[str, List[DocumentSection]]:
        """Process a PDF document."""
        logger.info("   → Processing PDF with pymupdf4llm...")

        # Extract markdown text
        markdown_text = pymupdf4llm.to_markdown(file_path)

        # Open document for structure extraction
        doc = pymupdf.open(file_path)
        toc = doc.get_toc()

        # Build sections from table of contents
        sections = self._build_sections_from_toc(doc, toc)

        logger.info(f"   → Extracted {len(sections)} sections from PDF")
        doc.close()

        return markdown_text, sections

    def _process_latex(self, file_path: str) -> Tuple[str, List[DocumentSection]]:
        """Process a LaTeX document."""
        logger.info("   → Processing LaTeX file...")

        with open(file_path, "r", encoding="utf-8") as f:
            latex_content = f.read()

        # Convert to text
        text_content = LatexNodes2Text().latex_to_text(latex_content)

        # Extract structure
        sections = self._extract_latex_structure(latex_content)

        logger.info(f"   → Extracted {len(sections)} sections from LaTeX")

        return text_content, sections

    def _build_sections_from_toc(
        self, doc: pymupdf.Document, toc: List[List[Any]]
    ) -> List[DocumentSection]:
        """Build document sections from PDF table of contents."""
        if not toc:
            # Fallback: create single section with all content
            full_text = ""
            for page in doc:
                full_text += page.get_text()

            return [
                DocumentSection(
                    title="Full Document",
                    content=full_text,
                    level=1,
                    page_start=0,
                    page_end=len(doc) - 1,
                    parent_titles=[],
                )
            ]

        sections = []
        for i, (level, title, page) in enumerate(toc):
            # Determine end page
            end_page = len(doc) - 1
            if i + 1 < len(toc):
                end_page = toc[i + 1][2] - 1

            # Extract content for this section
            content = ""
            for page_num in range(page - 1, min(end_page + 1, len(doc))):
                content += doc[page_num].get_text()

            # Determine parent titles (simplified - could be enhanced)
            parent_titles = []
            for j in range(i - 1, -1, -1):
                if toc[j][0] < level:
                    parent_titles.insert(0, toc[j][1])
                    if toc[j][0] == 1:  # Stop at chapter level
                        break

            sections.append(
                DocumentSection(
                    title=title,
                    content=content,
                    level=level,
                    page_start=page - 1,
                    page_end=end_page,
                    parent_titles=parent_titles,
                )
            )

        return sections

    def _extract_latex_structure(self, latex_content: str) -> List[DocumentSection]:
        """Extract structure from LaTeX content."""
        # Pattern to match LaTeX sectioning commands
        section_pattern = re.compile(
            r"\\(chapter|section|subsection|subsubsection)\*?\{([^}]+)\}", re.MULTILINE
        )

        matches = list(section_pattern.finditer(latex_content))
        if not matches:
            # No structure found, return entire content as one section
            return [
                DocumentSection(
                    title="Full Document",
                    content=latex_content,
                    level=1,
                    page_start=0,
                    page_end=0,
                    parent_titles=[],
                )
            ]

        sections = []
        level_map = {"chapter": 1, "section": 2, "subsection": 3, "subsubsection": 4}

        for i, match in enumerate(matches):
            command = match.group(1)
            title = match.group(2).strip()
            level = level_map.get(command, 99)

            # Extract content between this section and the next
            start_pos = match.end()
            end_pos = len(latex_content)
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()

            content = latex_content[start_pos:end_pos].strip()

            # Convert content to plain text
            try:
                content = LatexNodes2Text().latex_to_text(content)
            except:
                # If conversion fails, use raw content
                pass

            # Determine parent titles (simplified)
            parent_titles = []
            for j in range(i - 1, -1, -1):
                prev_level = level_map.get(matches[j].group(1), 99)
                if prev_level < level:
                    parent_titles.insert(0, matches[j].group(2).strip())
                    if prev_level == 1:  # Stop at chapter level
                        break

            sections.append(
                DocumentSection(
                    title=title,
                    content=content,
                    level=level,
                    page_start=0,  # Page numbers not available for LaTeX
                    page_end=0,
                    parent_titles=parent_titles,
                )
            )

        return sections

    async def _add_source_to_graph(self, metadata: DocumentMetadata) -> str:
        """Add source document to the graph."""
        source_id = metadata.title.lower().replace(" ", "-").replace("/", "-")

        source_data = {
            "id": source_id,
            "title": metadata.title,
            "authors": metadata.authors,
            "publication_year": metadata.publication_year,
            "document_type": metadata.document_type,
            "source_path": metadata.source_path,
        }

        if metadata.doi:
            source_data["doi"] = metadata.doi
        if metadata.isbn:
            source_data["isbn"] = metadata.isbn

        await self.graph_manager.add_source(source_data)
        logger.info(f"✅ Added source to graph: {source_id}")

        return source_id

    async def _execute_extraction_pass(
        self,
        pass_config,
        sections: List[DocumentSection],
        source_id: str,
        metadata: DocumentMetadata,
    ) -> int:
        """
        Execute a single extraction pass across all document sections.

        Args:
            pass_config: Configuration for this extraction pass
            sections: Document sections to process
            source_id: ID of the source document
            metadata: Document metadata

        Returns:
            Number of items extracted in this pass
        """
        total_items = 0

        # Build context for this pass if needed
        context_glossary = ""
        if pass_config.requires_context:
            if pass_config.context_entities:
                # Get context from specific entity types
                context_glossary = await self._build_context_glossary(
                    source_id, pass_config.context_entities
                )
            else:
                # Get general context from the graph
                context_glossary = await self.graph_manager.get_global_glossary()

        # Process each section
        for section in sections:
            if not section.content.strip():
                continue

            logger.info(f"   → Processing section: {section.title}")

            try:
                items_extracted = await self._extract_from_section(
                    pass_config, section, source_id, context_glossary
                )
                total_items += items_extracted

            except Exception as e:
                logger.error(f"   ❌ Failed to process section '{section.title}': {e}")
                continue

        return total_items

    async def _extract_from_section(
        self,
        pass_config,
        section: DocumentSection,
        source_id: str,
        context_glossary: str,
    ) -> int:
        """
        Extract entities from a single document section.

        Args:
            pass_config: Extraction pass configuration
            section: Document section to process
            source_id: Source document ID
            context_glossary: Context glossary for this pass

        Returns:
            Number of items extracted from this section
        """
        # Generate extraction prompt
        prompt = await self.prompt_engine.generate_pass_specific_prompt(
            pass_config, section.content, context_glossary
        )

        # Call LLM for extraction
        llm_response = await self.system.graphiti.llm_client.generate_response(
            messages=[Message(role="user", content=prompt)]
        )

        # Parse LLM response
        extracted_data = self._parse_llm_response(
            llm_response, pass_config.name, section.title
        )

        if not extracted_data:
            return 0

        # Process extracted entities
        entities = extracted_data.get("extracted_entities", [])
        if not entities:
            logger.info(f"      → No entities found in section: {section.title}")
            return 0

        # Add entities to graph
        items_processed = 0
        for entity_data in entities:
            try:
                entity_type = entity_data.get("type")
                if entity_type not in pass_config.target_entities:
                    logger.warning(
                        f"      ⚠️ Skipping unexpected entity type: {entity_type}"
                    )
                    continue

                # Add section context to entity
                entity_data["source_section"] = section.title
                entity_data["parent_sections"] = " > ".join(section.parent_titles)

                # Add entity to graph
                await self._add_entity_to_graph(entity_data, source_id)
                items_processed += 1

            except Exception as e:
                logger.error(
                    f"      ❌ Failed to add entity {entity_data.get('id', 'unknown')}: {e}"
                )
                continue

        logger.info(
            f"      ✅ Extracted {items_processed} entities from section: {section.title}"
        )
        return items_processed

    async def _add_entity_to_graph(
        self, entity_data: Dict[str, Any], source_id: str
    ) -> None:
        """
        Add an extracted entity to the graph with relationships.

        Args:
            entity_data: Entity data from LLM extraction
            source_id: Source document ID
        """
        entity_type = entity_data.pop("type")
        entity_id = entity_data.get("id")

        # Extract relationship data
        relationship_data = {}
        for key in list(entity_data.keys()):
            if key.startswith("uses_") or key in [
                "proves",
                "specializes",
                "generalizes",
            ]:
                relationship_data[key] = entity_data.pop(key)

        # Add entity to graph
        await self.graph_manager.add_entity(entity_type, entity_data, source_id)

        # Create relationships
        await self._create_entity_relationships(entity_id, relationship_data)

    async def _create_entity_relationships(
        self, entity_id: str, relationship_data: Dict[str, Any]
    ) -> None:
        """
        Create relationships for an entity based on extracted data.

        Args:
            entity_id: ID of the source entity
            relationship_data: Dictionary of relationship information
        """
        for rel_key, target_ids in relationship_data.items():
            if not target_ids:
                continue

            # Map relationship keys to relationship types
            rel_type_map = {
                "uses_definitions": "USES",
                "uses_facts": "USES",
                "proves": "PROVES",
                "specializes": "SPECIALIZES",
                "generalizes": "GENERALIZES",
            }

            rel_type = rel_type_map.get(rel_key)
            if not rel_type:
                logger.warning(f"Unknown relationship key: {rel_key}")
                continue

            # Ensure target_ids is a list
            if isinstance(target_ids, str):
                target_ids = [target_ids]

            # Create relationships
            for target_id in target_ids:
                try:
                    await self.graph_manager.create_relationship(
                        rel_type, entity_id, target_id, validate=False
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to create relationship {rel_type}: {entity_id} -> {target_id}: {e}"
                    )

    def _parse_llm_response(
        self, llm_response: Dict[str, Any], pass_name: str, section_title: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse and validate LLM response JSON.

        Args:
            llm_response: Response from LLM
            pass_name: Name of the extraction pass
            section_title: Title of the section being processed

        Returns:
            Parsed JSON data or None if parsing failed
        """
        try:
            content = llm_response.get("content", "")
            if not content:
                logger.warning(f"   ⚠️ Empty LLM response for {section_title}")
                return None

            # Extract JSON block
            json_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
            if not json_match:
                logger.warning(
                    f"   ⚠️ No JSON block found in response for {section_title}"
                )
                return None

            json_string = json_match.group(1)

            # Try standard JSON first, then demjson for robustness
            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                return demjson3.decode(json_string)

        except Exception as e:
            logger.error(f"   ❌ Failed to parse LLM response for {section_title}: {e}")
            return None

    async def _build_context_glossary(
        self, source_id: str, context_entities: List[str]
    ) -> str:
        """
        Build a context glossary for specific entity types.

        Args:
            source_id: Source document ID
            context_entities: List of entity types to include in context

        Returns:
            Formatted context glossary string
        """
        glossary_parts = []

        for entity_type in context_entities:
            entities = await self.graph_manager.find_entities(entity_type, limit=20)

            for entity in entities:
                if entity_type == "Definition":
                    term = entity.get("term", "")
                    definition = entity.get("definition", "")
                    if term and definition:
                        glossary_parts.append(f"- Definition: {term}\n  {definition}")

                elif entity_type == "Fact":
                    content = entity.get("content", "")
                    if content:
                        # Truncate long facts
                        display_content = (
                            content[:150] + "..." if len(content) > 150 else content
                        )
                        glossary_parts.append(f"- Fact: {display_content}")

        return "\n".join(glossary_parts)


class BulkIngestionManager:
    """Manages bulk ingestion of multiple documents."""

    def __init__(self, system, schema_registry: SchemaRegistry):
        """
        Initialize the bulk ingestion manager.

        Args:
            system: System instance
            schema_registry: Schema registry for configuration
        """
        self.system = system
        self.processor = DynamicDocumentProcessor(system, schema_registry)

    async def ingest_documents(self, document_configs: List[Dict[str, Any]]) -> None:
        """
        Ingest multiple documents with their configurations.

        Args:
            document_configs: List of document configurations
        """
        logger.info(
            f"📚 Starting bulk ingestion of {len(document_configs)} documents..."
        )

        total_items = 0

        for config in document_configs:
            file_path = config["file_path"]
            metadata = config["metadata"]

            try:
                items_extracted = await self.processor.process_document(
                    file_path, metadata
                )
                total_items += items_extracted
                logger.info(f"✅ Processed {metadata.title}: {items_extracted} items")

            except Exception as e:
                logger.error(f"❌ Failed to process {metadata.title}: {e}")
                continue

        logger.info(f"🎉 Bulk ingestion complete! Total items extracted: {total_items}")


# Example document configurations
EXAMPLE_DOCUMENT_CONFIGS = [
    {
        "file_path": "data/papers/LIPIcs.CSL.2011.174.pdf",
        "metadata": DocumentMetadata(
            title="A semantic approach to illative combinatory logic",
            authors=["Łukasz Czajka"],
            publication_year=2011,
            document_type="paper",
            source_path="data/papers/LIPIcs.CSL.2011.174.pdf",
            doi="10.4230/LIPIcs.CSL.2011.174",
        ),
    },
]


async def run_ingestion(document_configs: List[Dict[str, Any]] = None):
    """Run the complete ingestion pipeline."""
    from infrastructure_setup import initialize_system

    # Initialize system
    system = await initialize_system()

    # Create schema registry
    from core.schema_registry import SchemaRegistry

    schema_registry = SchemaRegistry("config/graph_schema.yaml")

    # Create ingestion manager
    ingestion_manager = BulkIngestionManager(system, schema_registry)

    # Use provided configs or examples
    configs = document_configs or EXAMPLE_DOCUMENT_CONFIGS

    # Run ingestion
    await ingestion_manager.ingest_documents(configs)

    logger.info("🎉 Ingestion pipeline complete!")


if __name__ == "__main__":
    asyncio.run(run_ingestion())
