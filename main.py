# main.py
"""
Main CLI application for the schema-driven combinatory logic assistant.
Provides an interactive interface with comprehensive system integration.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from infrastructure_setup import initialize_system
from multi_turn_agent import CombinatoryLogicAgent, EnhancedAgentOrchestrator
from processing.dynamic_document_processor import BulkIngestionManager, DocumentMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CombinatoryLogicCLI:
    """
    Enhanced command-line interface for the combinatory logic assistant.
    Supports both interactive queries and administrative operations.
    """

    def __init__(self, config_path: str = "config/graph_schema.yaml"):
        """
        Initialize the CLI with schema configuration.

        Args:
            config_path: Path to the graph schema configuration
        """
        self.config_path = config_path
        self.system = None
        self.agent_orchestrator = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize the system and agent components."""
        try:
            logger.info("🚀 Initializing Combinatory Logic Assistant...")

            # Initialize the complete system
            self.system = await initialize_system(self.config_path)

            # Initialize agent orchestrator
            self.agent_orchestrator = EnhancedAgentOrchestrator(self.system)

            self.is_initialized = True
            logger.info("✅ Assistant is ready!")

        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise

    async def run_interactive(self):
        """Run the main interactive CLI loop."""
        if not self.is_initialized:
            await self.initialize()

        print("\n🧠 Combinatory Logic Assistant")
        print("=" * 50)
        print("Schema-driven RAG system for mathematical logic")
        print("Commands:")
        print("  - Type your questions normally")
        print("  - '/help' for available commands")
        print("  - '/status' for system status")
        print("  - '/schema' for schema information")
        print("  - '/ingest <file>' to ingest documents")
        print("  - '/quit' to exit")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n💭 You: ").strip()

                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Regular query processing
                print("\n🤖 Assistant: ", end="", flush=True)
                response = await self.agent_orchestrator.route_query(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\n❌ Error: {e}")

        await self.close()

    async def _handle_command(self, command: str):
        """Handle special CLI commands."""
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/quit" or cmd == "/exit":
            print("\n👋 Goodbye!")
            await self.close()
            sys.exit(0)

        elif cmd == "/help":
            await self._show_help()

        elif cmd == "/status":
            await self._show_status()

        elif cmd == "/schema":
            await self._show_schema()

        elif cmd == "/ingest":
            if len(parts) > 1:
                await self._ingest_document(parts[1])
            else:
                print("❌ Usage: /ingest <file_path>")

        elif cmd == "/search":
            if len(parts) > 1:
                query = " ".join(parts[1:])
                await self._quick_search(query)
            else:
                print("❌ Usage: /search <query>")

        elif cmd == "/entities":
            await self._list_entities()

        elif cmd == "/sources":
            await self._list_sources()

        else:
            print(f"❌ Unknown command: {cmd}. Type '/help' for available commands.")

    async def _show_help(self):
        """Show available commands and usage."""
        help_text = """
📖 Available Commands:

🔍 **Information Commands:**
  /status    - Show system status and statistics
  /schema    - Display knowledge graph schema information
  /entities  - List available entity types and counts
  /sources   - List all source documents

🔧 **Data Management:**
  /ingest <file>    - Ingest a document into the knowledge graph
  /search <query>   - Quick search across the knowledge graph

🎯 **Utility Commands:**
  /help      - Show this help message
  /quit      - Exit the application

💡 **Tips:**
  - Ask questions naturally about combinatory logic
  - Request specific entity types: "Show me all definitions of..."
  - Ask for proofs: "How is theorem X proven?"
  - Request comparisons: "What's the difference between X and Y?"
  - Ask for textbook content: "Write a section about..."
"""
        print(help_text)

    async def _show_status(self):
        """Show current system status."""
        try:
            print("\n📊 System Status:")
            print("-" * 30)

            # Entity counts
            entity_types = self.system.schema_registry.get_all_entity_types()
            for entity_type in entity_types:
                count = await self.system.graph_manager.count_entities(entity_type)
                print(f"  {entity_type}: {count}")

            # Total entities
            total_count = await self.system.graph_manager.count_entities()
            print(f"\n  Total Entities: {total_count}")

            # Schema information
            schema_summary = self.system.schema_registry.export_schema_summary()
            print(f"\n📋 Schema Info:")
            print(f"  Entity Types: {len(schema_summary.get('entities', {}))}")
            print(
                f"  Relationship Types: {len(schema_summary.get('relationships', {}))}"
            )
            print(f"  Extraction Passes: {schema_summary.get('extraction_passes', 0)}")

        except Exception as e:
            print(f"❌ Error getting status: {e}")

    async def _show_schema(self):
        """Show schema information."""
        try:
            print("\n📋 Knowledge Graph Schema:")
            print("-" * 40)

            schema_summary = self.system.schema_registry.export_schema_summary()

            print("\n🏷️ Entity Types:")
            for name, info in schema_summary.get("entities", {}).items():
                print(f"  • {name}: {info['description']}")
                print(f"    Properties: {info['properties']}")
                print(
                    f"    Outgoing: {', '.join(info.get('outgoing_relationships', []))}"
                )
                print()

            print("\n🔗 Relationship Types:")
            for name, info in schema_summary.get("relationships", {}).items():
                print(f"  • {name}: {info['description']}")
                print(
                    f"    {' → '.join(info['source_entities'])} → {' → '.join(info['target_entities'])}"
                )
                print()

        except Exception as e:
            print(f"❌ Error showing schema: {e}")

    async def _ingest_document(self, file_path: str):
        """Ingest a document into the knowledge graph."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return

            print(f"📄 Ingesting document: {file_path}")

            # Create document metadata (simplified - could be enhanced with input prompts)
            metadata = DocumentMetadata(
                title=file_path.stem.replace("_", " ").replace("-", " ").title(),
                authors=["Unknown"],
                publication_year=2024,
                document_type="document",
                source_path=str(file_path),
            )

            # Initialize ingestion manager
            ingestion_manager = BulkIngestionManager(
                self.system, self.system.schema_registry
            )

            # Ingest the document
            document_configs = [{"file_path": str(file_path), "metadata": metadata}]

            await ingestion_manager.ingest_documents(document_configs)
            print("✅ Document ingestion completed!")

        except Exception as e:
            print(f"❌ Error ingesting document: {e}")

    async def _quick_search(self, query: str):
        """Perform a quick search and display results."""
        try:
            print(f"\n🔍 Searching for: '{query}'")

            # Use the agent's search capabilities
            from advanced_retrieval import SchemaAwareKnowledgeGraphTool

            kg_tool = SchemaAwareKnowledgeGraphTool(self.system)
            results = await kg_tool.search(query, limit=5)

            if results.nodes:
                print(f"\n✅ Found {len(results.nodes)} results:")
                for i, node in enumerate(results.nodes, 1):
                    content = node.get(
                        "content", node.get("definition", str(node)[:100])
                    )
                    print(f"  {i}. {content[:150]}...")
            else:
                print("❌ No results found.")

        except Exception as e:
            print(f"❌ Search error: {e}")

    async def _list_entities(self):
        """List entity types and their counts."""
        try:
            print("\n🏷️ Entity Types and Counts:")
            print("-" * 30)

            entity_types = self.system.schema_registry.get_all_entity_types()
            for entity_type in sorted(entity_types):
                count = await self.system.graph_manager.count_entities(entity_type)
                schema = self.system.schema_registry.get_entity_schema(entity_type)
                description = schema.description if schema else "No description"
                print(f"  {entity_type}: {count}")
                print(f"    {description}")
                print()

        except Exception as e:
            print(f"❌ Error listing entities: {e}")

    async def _list_sources(self):
        """List all source documents."""
        try:
            print("\n📚 Source Documents:")
            print("-" * 25)

            sources = await self.system.graph_manager.find_entities("Source", limit=50)

            if sources:
                for source in sources:
                    title = source.get("title", "Unknown Title")
                    authors = source.get("authors", ["Unknown Author"])
                    year = source.get("publication_year", "Unknown Year")
                    doc_type = source.get("document_type", "Unknown Type")

                    print(f"  📖 {title}")
                    print(
                        f"     Authors: {', '.join(authors) if isinstance(authors, list) else authors}"
                    )
                    print(f"     Year: {year} | Type: {doc_type}")
                    print()
            else:
                print("  No sources found.")

        except Exception as e:
            print(f"❌ Error listing sources: {e}")

    async def close(self):
        """Close system resources."""
        logger.info("🔄 Closing CLI resources...")

        if self.agent_orchestrator:
            await self.agent_orchestrator.close()

        if self.system:
            await self.system.close()

        logger.info("✅ CLI resources closed")


async def run_cli():
    """Main entry point for the CLI application."""
    try:
        cli = CombinatoryLogicCLI()
        await cli.run_interactive()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"❌ CLI error: {e}", exc_info=True)
        print(f"\n💥 Fatal error: {e}")


async def run_batch_ingestion(file_paths: list):
    """Run batch document ingestion."""
    try:
        logger.info(f"📚 Starting batch ingestion of {len(file_paths)} documents...")

        # Initialize system
        system = await initialize_system()

        # Prepare document configurations
        document_configs = []
        for file_path in file_paths:
            file_path = Path(file_path)
            if file_path.exists():
                metadata = DocumentMetadata(
                    title=file_path.stem.replace("_", " ").replace("-", " ").title(),
                    authors=["Unknown"],
                    publication_year=2024,
                    document_type="document",
                    source_path=str(file_path),
                )

                document_configs.append(
                    {"file_path": str(file_path), "metadata": metadata}
                )
            else:
                logger.warning(f"⚠️ File not found: {file_path}")

        if document_configs:
            # Run ingestion
            ingestion_manager = BulkIngestionManager(system, system.schema_registry)
            await ingestion_manager.ingest_documents(document_configs)
            logger.info("🎉 Batch ingestion completed!")
        else:
            logger.error("❌ No valid files found for ingestion")

        await system.close()

    except Exception as e:
        logger.error(f"❌ Batch ingestion failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Batch mode - ingest provided files
        file_paths = sys.argv[1:]
        asyncio.run(run_batch_ingestion(file_paths))
    else:
        # Interactive mode
        asyncio.run(run_cli())
