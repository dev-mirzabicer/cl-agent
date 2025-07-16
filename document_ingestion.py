import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json
import re
from dataclasses import dataclass
import demjson3

import pymupdf
import pymupdf4llm
from pylatexenc.latex2text import LatexNodes2Text
from typing import Tuple

from graphiti_core.nodes import EpisodeType
from graphiti_core.prompts.models import Message
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode


from graph_manager import GraphManager


@dataclass
class DocumentStructure:
    """Represents the hierarchical structure of a document."""

    title: str
    sections: List[Dict[str, Any]]  # Can be nested


@dataclass
class DocumentMetadata:
    """Metadata for processed documents"""

    title: str
    authors: List[str]
    publication_year: int
    document_type: str  # "book", "paper", "chapter"
    source_path: str
    page_range: Optional[tuple] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    structure: Optional[DocumentStructure] = None


class DocumentProcessor:
    """
    Processes PDF documents for ingestion into the knowledge graph
    """

    def __init__(self, system):
        self.system = system
        self.graphiti = system.graphiti
        self.graph_manager = GraphManager(system)

    async def process_document(self, file_path: str, metadata: DocumentMetadata) -> int:
        """
        Process a single document and return episodes for ingestion
        """
        print(f"📄 Processing {metadata.title}...")

        # Determine processing strategy based on document type
        if file_path.endswith(".pdf"):
            print("   -> Using PDF processing pipeline")
            text_content, structure = self._process_pdf(file_path)
        elif file_path.endswith(".tex"):
            print("   -> Using LaTeX processing pipeline")
            text_content, structure = self._process_latex(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        metadata.structure = structure

        # --- Multi-Pass Ingestion ---
        # Pass 1: Build the glossary
        print("\n--- Starting Pass 1: Glossary Extraction ---")
        num_glossary_items = await self._run_pass1_glossary_extraction(
            text_content, metadata
        )
        print(f"--- Pass 1 Complete: {num_glossary_items} glossary items found ---\n")

        # Pass 2: Extract context-aware entities
        print("--- Starting Pass 2: Statement & Argument Extraction ---")
        num_entity_items = await self._run_pass2_entity_extraction(
            text_content, metadata
        )
        print(f"--- Pass 2 Complete: {num_entity_items} entities found ---\n")

        return num_glossary_items + num_entity_items

    def _process_pdf(self, file_path: str) -> Tuple[str, DocumentStructure]:
        """Process a PDF document."""
        print("      - Extracting text with pymupdf4llm...")
        markdown_text = pymupdf4llm.to_markdown(file_path)
        doc = pymupdf.open(file_path)
        title = doc.metadata.get("title", "Unknown PDF")
        print("      - Extracting table of contents...")
        toc = doc.get_toc()
        structure = self._build_structure_from_toc(toc, doc_title=title)
        print(f"      - Found {len(toc)} top-level sections in ToC")

        return markdown_text, structure

    def _process_latex(self, file_path: str) -> Tuple[str, DocumentStructure]:
        """Process a LaTeX document and extract its structure."""
        print("      - Reading LaTeX file...")
        with open(file_path, "r", encoding="utf-8") as f:
            latex_content = f.read()

        print("      - Converting LaTeX to text...")
        # This is a basic conversion; for chunking, we primarily need the structure
        text_content = LatexNodes2Text().latex_to_text(latex_content)

        print("      - Extracting structure from LaTeX...")
        # Regex to find structure commands like \chapter, \section, \subsection
        structure_regex = re.compile(
            r"\\(chapter|section|subsection|subsubsection)\*?\{(.+?)\}", re.DOTALL
        )

        sections = []
        title = "LaTeX Document"
        title_match = re.search(r"\\title\{(.+?)\}", latex_content)
        if title_match:
            title = title_match.group(1).strip()

        for match in structure_regex.finditer(latex_content):
            level_map = {
                "chapter": 1,
                "section": 2,
                "subsection": 3,
                "subsubsection": 4,
            }
            command = match.group(1)
            section_title = match.group(2).strip()
            level = level_map.get(command, 99)
            # Page number is not available from static analysis
            sections.append({"title": section_title, "level": level, "page": -1})

        # Build a hierarchical structure from the flat list of sections
        hierarchical_sections: List[Dict[str, Any]] = []
        parent_stack: List[Dict[str, Any]] = [
            {"level": 0, "subsections": hierarchical_sections}
        ]

        for section in sections:
            level = int(section["level"])
            section_item: Dict[str, Any] = {
                "title": section["title"],
                "level": level,
                "page": -1,
                "subsections": [],
            }

            while level <= parent_stack[-1]["level"]:
                parent_stack.pop()

            parent_stack[-1]["subsections"].append(section_item)
            parent_stack.append(section_item)

        structure = DocumentStructure(title=title, sections=hierarchical_sections)
        print(f"      - Found {len(sections)} structural elements in LaTeX file.")

        return text_content, structure

    def _build_structure_from_toc(
        self, toc: List[List[Any]], doc_title: str = "PDF Document"
    ) -> DocumentStructure:
        """Build a hierarchical structure from a PyMuPDF TOC."""
        if not toc:
            print("      - No ToC found.")
            return DocumentStructure(title=doc_title, sections=[])

        sections: List[Dict[str, Any]] = []
        # A stack to keep track of the current path in the hierarchy
        parent_stack: List[Dict[str, Any]] = [{"level": 0, "subsections": sections}]

        for level, title, page in toc:
            section_item = {
                "title": title,
                "level": level,
                "page": page,
                "subsections": [],
            }

            # Go up the stack to find the correct parent for the current section
            while level <= parent_stack[-1]["level"]:
                parent_stack.pop()

            # Add the current section to its parent's subsections
            parent_stack[-1]["subsections"].append(section_item)
            # Push the current section onto the stack to become a potential parent
            parent_stack.append(section_item)

        return DocumentStructure(title=doc_title, sections=sections)

    def _extract_json_from_llm_response(
        self, llm_response: Dict[str, Any], pass_name: str, source_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts and parses the JSON block from an LLM response with robust error handling.
        """
        try:
            content = llm_response.get("content", "")
            if not content:
                print(f"   - {pass_name}: LLM response for {source_id} is empty.")
                return None

            json_string_match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
            if not json_string_match:
                print(
                    f"   - {pass_name}: No JSON block found in LLM response for {source_id}."
                )
                print("     LLM Response Content:")
                print(content)
                return None

            json_string = json_string_match.group(1)
            try:
                # Use demjson for more lenient parsing
                return demjson3.decode(json_string)
            except demjson3.JSONDecodeError as e:
                print(
                    f"   - {pass_name}: Failed to decode JSON for {source_id} even with demjson. Error: {e}"
                )
                print("     Problematic JSON string:")
                print(json_string)
                return None

        except Exception as e:
            print(
                f"   - {pass_name}: An unexpected error occurred while parsing LLM response for {source_id}: {e}"
            )
            return None

    async def _pass1_extract_glossary(
        self, chunk: str, source_id: str, section_title: str
    ) -> int:
        """
        Pass 1: Extracts Term and Symbol definitions from a text chunk and adds them
        to the graph as a foundational glossary.
        """
        from infrastructure_setup import TERM_EXTRACTION_PROMPT
        import uuid

        print(f"         - Pass 1: Analyzing chunk for glossary: '{section_title}'")
        prompt = TERM_EXTRACTION_PROMPT.replace("{text}", chunk)
        llm_response = await self.system.graphiti.llm_client.generate_response(
            messages=[Message(role="user", content=prompt)]
        )

        data = self._extract_json_from_llm_response(llm_response, "Pass 1", source_id)

        if not data:
            return 0

        items = data.get("glossary_items", [])
        if not items:
            print(
                f"         - Pass 1: No glossary items found in chunk '{section_title}'."
            )
            return 0

        print(
            f"         - Pass 1: Found {len(items)} glossary items in chunk '{section_title}'."
        )
        await self._process_glossary_items(items, source_id)
        return len(items)

    async def _process_glossary_items(
        self, items: List[Dict[str, Any]], source_id: str
    ):
        """Process and add a list of glossary items to the graph."""
        for item in items:
            print(
                f"           - Processing glossary item with keys: {list(item.keys())}"
            )
            item["source_id"] = source_id
            item_type = item.get("type")

            if item_type == "Term":
                await self.graph_manager.add_term(item)
            elif item_type == "Symbol":
                await self.graph_manager.add_symbol(item)
            else:
                print(f"           - WARNING: Unknown glossary item type '{item_type}'")

    async def _pass2_extract_entities(
        self,
        chunk: str,
        source_id: str,
        glossary: str,
        section_title: Optional[str] = None,
    ) -> int:
        """
        Pass 2: Extracts Statement and Argument entities from a text chunk, using the
        pre-built glossary for context.
        """
        from infrastructure_setup import STATEMENT_EXTRACTION_PROMPT
        import uuid

        print(f"         - Pass 2: Analyzing chunk for entities: '{section_title}'")
        prompt = STATEMENT_EXTRACTION_PROMPT.replace("{text}", chunk).replace(
            "{glossary}", glossary
        )
        llm_response = await self.system.graphiti.llm_client.generate_response(
            messages=[Message(role="user", content=prompt)]
        )

        data = self._extract_json_from_llm_response(llm_response, "Pass 2", source_id)

        if not data:
            return 0

        items = data.get("entities", [])
        if not items:
            print(f"         - Pass 2: No entities found in chunk '{section_title}'.")
            return 0

        print(
            f"         - Pass 2: Found {len(items)} entities in chunk '{section_title}'."
        )
        await self._process_entity_items(items, source_id)
        return len(items)

    async def _process_entity_items(self, items: List[Dict[str, Any]], source_id: str):
        """Process and add a list of entity items to the graph."""
        for item in items:
            print(f"           - Processing entity with keys: {list(item.keys())}")
            item["source_id"] = source_id
            item_type = item.get("type")

            if item_type == "Statement":
                await self.graph_manager.add_statement(item)
            elif item_type == "Argument":
                await self.graph_manager.add_argument(item)
            else:
                print(f"           - WARNING: Unknown entity type '{item_type}'")

    async def _process_sections_recursively(
        self,
        sections: List[Dict[str, Any]],
        doc: pymupdf.Document,
        metadata: DocumentMetadata,
        parent_titles: List[str],
        pass_function,
        **kwargs,
    ) -> int:
        """
        Recursively process sections and their subsections, applying the specified
        pass_function to each chunk.
        """
        total_items = 0
        for i, section in enumerate(sections):
            current_titles = parent_titles + [section["title"]]
            full_section_title = " > ".join(current_titles)

            start_page = section["page"] - 1
            end_page = len(doc)  # Default to end of doc
            if section["subsections"]:
                end_page = section["subsections"][0]["page"] - 1
            elif i + 1 < len(sections):
                # Find the next sibling at the same level to determine the end page
                next_sibling_page = len(doc)
                # This logic needs to be more robust to find the true end of a section
                # For now, we'll approximate by going to the start of the next top-level section if it exists
                if i + 1 < len(sections):
                    next_sibling_page = sections[i + 1]["page"] - 1
                end_page = next_sibling_page

            if start_page >= end_page:
                end_page = start_page + 1  # Process at least one page

            chunk = ""
            for page_num in range(start_page, min(end_page, len(doc))):
                chunk += doc[page_num].get_text()

            if chunk.strip():
                print(
                    f"         - Processing chunk for section: '{full_section_title}' (pages {start_page + 1}-{end_page})"
                )
                num_items = await pass_function(
                    chunk=chunk,
                    source_id=metadata.title.lower().replace(" ", "-"),
                    section_title=full_section_title,
                    **kwargs,
                )
                total_items += num_items

            if section["subsections"]:
                total_items += await self._process_sections_recursively(
                    section["subsections"],
                    doc,
                    metadata,
                    current_titles,
                    pass_function,
                    **kwargs,
                )
        return total_items

    async def _run_pass2_entity_extraction(
        self, markdown_text: str, metadata: DocumentMetadata
    ) -> int:
        """
        Orchestrates the second pass of the ingestion pipeline: statement and
        argument extraction.
        """
        print("   -> Segmenting document for entity extraction...")
        source_id = metadata.title.lower().replace(" ", "-")
        print("   -> Fetching global glossary for context-aware extraction...")
        glossary = await self.graph_manager.get_global_glossary()
        if not glossary:
            print("      - Warning: Global glossary is empty. Context will be limited.")

        if (
            metadata.source_path.endswith(".pdf")
            and metadata.structure
            and metadata.structure.sections
        ):
            doc = pymupdf.open(metadata.source_path)
            return await self._process_sections_recursively(
                sections=metadata.structure.sections,
                doc=doc,
                metadata=metadata,
                parent_titles=[],
                pass_function=self._pass2_extract_entities,
                glossary=glossary,
            )
        else:
            # Fallback for non-PDFs or PDFs without structure
            # This part remains the same as the old fallback
            total_items = 0
            chunks = re.split(r"(^## .*)", markdown_text, flags=re.MULTILINE)
            processed_chunks = []
            if chunks[0].strip():
                processed_chunks.append(("Introduction", chunks[0]))
            for i in range(1, len(chunks), 2):
                header = chunks[i].strip().lstrip("## ").strip()
                content = chunks[i + 1] if (i + 1) < len(chunks) else ""
                processed_chunks.append((header, content))
            for i, (section_title, chunk_content) in enumerate(processed_chunks):
                if not chunk_content.strip():
                    continue
                num_items = await self._pass2_extract_entities(
                    chunk=chunk_content, source_id=source_id, glossary=glossary
                )
                total_items += num_items
            return total_items

    async def _run_pass1_glossary_extraction(
        self, markdown_text: str, metadata: DocumentMetadata
    ) -> int:
        """
        Orchestrates the first pass of the ingestion pipeline: glossary extraction.
        """
        print("   -> Segmenting document for glossary extraction...")
        source_id = metadata.title.lower().replace(" ", "-")
        await self.graph_manager.add_source(
            {
                "id": source_id,
                "title": metadata.title,
                "authors": metadata.authors,
                "publication_year": metadata.publication_year,
                "source_path": metadata.source_path,
            }
        )

        if (
            metadata.source_path.endswith(".pdf")
            and metadata.structure
            and metadata.structure.sections
        ):
            doc = pymupdf.open(metadata.source_path)
            return await self._process_sections_recursively(
                sections=metadata.structure.sections,
                doc=doc,
                metadata=metadata,
                parent_titles=[],
                pass_function=self._pass1_extract_glossary,
            )
        else:
            # Fallback for non-PDFs or PDFs without structure
            total_items = 0
            chunks = re.split(r"(^## .*)", markdown_text, flags=re.MULTILINE)
            processed_chunks = []
            if chunks[0].strip():
                processed_chunks.append(("Introduction", chunks[0]))
            for i in range(1, len(chunks), 2):
                header = chunks[i].strip().lstrip("## ").strip()
                content = chunks[i + 1] if (i + 1) < len(chunks) else ""
                processed_chunks.append((header, content))
            for i, (section_title, chunk_content) in enumerate(processed_chunks):
                if not chunk_content.strip():
                    continue
                num_items = await self._pass1_extract_glossary(
                    chunk=chunk_content,
                    source_id=source_id,
                    section_title=section_title,
                )
                total_items += num_items
            return total_items


class BulkIngestionManager:
    """
    Manages bulk ingestion of multiple documents
    """

    def __init__(self, system):
        self.system = system
        self.processor = DocumentProcessor(system)

    async def ingest_documents(self, document_configs: List[Dict[str, Any]]) -> None:
        """
        Ingest multiple documents with their configurations

        Args:
            document_configs: List of dicts with keys:
                - file_path: Path to the PDF file
                - metadata: DocumentMetadata instance
                - group_id: Group identifier for organizing documents
        """

        print(f"📚 Starting ingestion of {len(document_configs)} documents...")

        all_episodes: List[RawEpisode] = []

        # Process each document
        for config in document_configs:
            file_path = config["file_path"]
            metadata = config["metadata"]
            group_id = config.get("group_id", "combinatory-logic")

            # Process document into episodes
            num_items = await self.processor.process_document(file_path, metadata)

            print(f"✓ Processed {metadata.title}: {num_items} items")

        # The new ingestion logic will be implemented in Phase 2
        print(
            f"✅ Document processing complete. Ingestion logic will be built in the next phase."
        )


# Example usage configuration
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
        "group_id": "combinatory-logic-papers",
    },
]


# Ingestion script
async def run_ingestion():
    """Run the complete ingestion pipeline"""

    # Initialize system
    from infrastructure_setup import initialize_system

    system = await initialize_system()

    # Create ingestion manager
    ingestion_manager = BulkIngestionManager(system)

    # Run ingestion
    await ingestion_manager.ingest_documents(EXAMPLE_DOCUMENT_CONFIGS)

    print("🎉 Ingestion complete! Knowledge graph is ready for querying.")


if __name__ == "__main__":
    asyncio.run(run_ingestion())
