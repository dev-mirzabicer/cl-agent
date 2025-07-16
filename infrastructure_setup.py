# infrastructure_setup.py
"""
Infrastructure setup for the schema-driven combinatory logic RAG system.
Initializes all components with centralized configuration management.
"""

import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

from core.schema_registry import SchemaRegistry
from core.dynamic_graph_manager import DynamicGraphManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class CombinatoryLogicRAGSystem:
    """
    Main system class that orchestrates the entire schema-driven RAG pipeline.
    Provides centralized management of all components.
    """

    def __init__(self, config_path: str = "config/graph_schema.yaml"):
        """
        Initialize the RAG system.

        Args:
            config_path: Path to the graph schema configuration file
        """
        self.config_path = config_path
        self.schema_registry = None
        self.graphiti = None
        self.graph_manager = None

        # Initialize components
        self._setup_schema_registry()
        self._setup_graphiti()
        self._setup_graph_manager()

    def _setup_schema_registry(self):
        """Initialize the schema registry with configuration."""
        logger.info("🔧 Setting up schema registry...")

        # Ensure config directory exists
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(exist_ok=True)

        # Create default config if it doesn't exist
        if not Path(self.config_path).exists():
            self._create_default_config()

        # Initialize schema registry
        self.schema_registry = SchemaRegistry(self.config_path)
        logger.info("✅ Schema registry initialized")

    def _create_default_config(self):
        """Create a default configuration file if none exists."""
        logger.info(f"📝 Creating default configuration at {self.config_path}")

        # This would copy the YAML content from our schema config artifact
        # For now, we assume the config file exists
        default_config = """
# Default minimal configuration - replace with full config
entities:
  Fact:
    description: "Mathematical facts and theorems"
    llm_instructions: "Extract mathematical statements that can be proven true or false"
    properties:
      id: {type: string, required: true, unique: true}
      content: {type: text, required: true, indexed: true}
    relationships:
      outgoing: [USES]
      incoming: [PART_OF]

relationships:
  USES:
    description: "Entity uses another entity"
    llm_instructions: "Use when one entity references another"
    source_entities: [Fact]
    target_entities: [Fact]

extraction_strategy:
  passes:
    - name: "Basic Extraction"
      target_entities: [Fact]
      context_entities: []
      requires_context: false

neo4j_setup:
  constraints: []
  indexes: []
"""

        with open(self.config_path, "w") as f:
            f.write(default_config)

    def _setup_graphiti(self):
        """Initialize Graphiti with optimized settings for mathematical content."""
        logger.info("🔧 Setting up Graphiti...")

        # Validate required environment variables
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        neo4j_uri = os.getenv("NEO4J_URI")
        neo4j_user = os.getenv("NEO4J_USER")
        neo4j_password = os.getenv("NEO4J_PASSWORD")

        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            raise ValueError("Neo4j connection environment variables not set")

        # Configure LLM for entity extraction
        llm_client = GeminiClient(
            config=LLMConfig(api_key=api_key, model="gemini-2.5-flash")
        )

        # Configure embedder for mathematical content
        embedder = GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=api_key, embedding_model="gemini-embedding-exp-03-07"
            )
        )

        # Configure cross-encoder for reranking
        cross_encoder = GeminiRerankerClient(
            config=LLMConfig(
                api_key=api_key, model="gemini-2.5-flash-lite-preview-06-17"
            )
        )

        # Initialize Graphiti
        self.graphiti = Graphiti(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

        logger.info("✅ Graphiti initialized")

    def _setup_graph_manager(self):
        """Initialize the dynamic graph manager."""
        logger.info("🔧 Setting up dynamic graph manager...")

        self.graph_manager = DynamicGraphManager(
            self.graphiti.driver, self.schema_registry
        )

        logger.info("✅ Dynamic graph manager initialized")

    async def setup_database_schema(self):
        """Setup database schema based on configuration."""
        logger.info("🔧 Setting up database schema...")

        # Build Graphiti indices and constraints
        await self.graphiti.build_indices_and_constraints()
        logger.info("✅ Graphiti indices and constraints created")

        # Setup schema-specific constraints and indexes
        await self.graph_manager.setup_database_schema()
        logger.info("✅ Schema-specific database setup complete")

    async def validate_system(self):
        """Validate that all system components are working correctly."""
        logger.info("🔍 Validating system components...")

        try:
            # Test database connection
            count = await self.graph_manager.count_entities()
            logger.info(f"✅ Database connection verified ({count} entities in graph)")

            # Test schema registry
            entity_types = self.schema_registry.get_all_entity_types()
            rel_types = self.schema_registry.get_all_relationship_types()
            logger.info(
                f"✅ Schema registry verified ({len(entity_types)} entities, {len(rel_types)} relationships)"
            )

            # Test LLM connection
            from graphiti_core.prompts.models import Message

            test_response = await self.graphiti.llm_client.generate_response(
                messages=[
                    Message(
                        role="user", content="Hello, can you respond with just 'OK'?"
                    )
                ]
            )
            if test_response and test_response.get("content"):
                logger.info("✅ LLM connection verified")
            else:
                logger.warning("⚠️ LLM connection test failed")

            logger.info("🎉 System validation complete - all components operational")

        except Exception as e:
            logger.error(f"❌ System validation failed: {e}")
            raise

    def get_schema_summary(self) -> dict:
        """Get a summary of the current schema configuration."""
        return self.schema_registry.export_schema_summary()

    async def close(self):
        """Close system resources."""
        logger.info("🔄 Closing system resources...")

        if self.graphiti:
            await self.graphiti.close()

        logger.info("✅ System resources closed")


async def initialize_system(
    config_path: str = "config/graph_schema.yaml",
) -> CombinatoryLogicRAGSystem:
    """
    Initialize the complete RAG system with all components.

    Args:
        config_path: Path to the schema configuration file

    Returns:
        Initialized RAG system instance
    """
    logger.info("🚀 Initializing Combinatory Logic RAG System...")

    try:
        # Create system instance
        system = CombinatoryLogicRAGSystem(config_path)

        # Setup database schema
        await system.setup_database_schema()

        # Validate system components
        await system.validate_system()

        logger.info("🎉 System initialization complete!")
        return system

    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise


async def create_test_entities(system: CombinatoryLogicRAGSystem):
    """
    Create some test entities to verify the system is working.
    This is useful for development and testing.
    """
    logger.info("🧪 Creating test entities...")

    try:
        # Add a test source
        source_id = await system.graph_manager.add_source(
            {
                "id": "test_source",
                "title": "Test Document",
                "authors": ["Test Author"],
                "publication_year": 2024,
                "document_type": "test",
                "source_path": "/test/path",
            }
        )

        # Add a test definition
        definition_id = await system.graph_manager.add_entity(
            "Definition",
            {
                "id": "test_definition",
                "term": "Test Term",
                "definition": "A term used for testing the system",
                "informal_explanation": "This is just a test definition",
            },
            source_id,
        )

        # Add a test fact
        fact_id = await system.graph_manager.add_entity(
            "Fact",
            {
                "id": "test_fact",
                "content": "This is a test mathematical fact",
                "explanation": "A fact created to test the system",
                "statement_type": "test",
            },
            source_id,
        )

        # Create a test relationship
        await system.graph_manager.create_relationship("USES", fact_id, definition_id)

        logger.info("✅ Test entities created successfully")

        # Verify we can retrieve them
        retrieved_fact = await system.graph_manager.get_entity(fact_id)
        if retrieved_fact:
            logger.info(f"✅ Test entity retrieval successful: {retrieved_fact['id']}")

        relationships = await system.graph_manager.get_entity_relationships(fact_id)
        if relationships:
            logger.info(
                f"✅ Test relationship retrieval successful: {len(relationships)} relationships"
            )

    except Exception as e:
        logger.error(f"❌ Test entity creation failed: {e}")
        raise


async def cleanup_test_entities(system: CombinatoryLogicRAGSystem):
    """Clean up test entities from the database."""
    logger.info("🧹 Cleaning up test entities...")

    try:
        # Delete test entities
        test_entities = ["test_fact", "test_definition", "test_source"]
        for entity_id in test_entities:
            await system.graph_manager.delete_entity(entity_id)

        logger.info("✅ Test entities cleaned up")

    except Exception as e:
        logger.error(f"❌ Test cleanup failed: {e}")


# Development and testing utilities
async def run_system_test():
    """Run a complete system test."""
    logger.info("🔬 Running complete system test...")

    system = None
    try:
        # Initialize system
        system = await initialize_system()

        # Create test entities
        await create_test_entities(system)

        # Test search functionality
        search_results = await system.graph_manager.search_entities_fulltext(
            ["Fact", "Definition"], "test", limit=10
        )
        logger.info(f"✅ Search test successful: {len(search_results)} results")

        # Test glossary building
        glossary = await system.graph_manager.get_global_glossary()
        logger.info(f"✅ Glossary test successful: {len(glossary)} characters")

        # Clean up
        await cleanup_test_entities(system)

        logger.info("🎉 System test completed successfully!")

    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        raise
    finally:
        if system:
            await system.close()


if __name__ == "__main__":
    # Run system test when executed directly
    asyncio.run(run_system_test())
