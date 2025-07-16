import os
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from neo4j import GraphDatabase

load_dotenv()


class CombinatoryLogicRAGSystem:
    """
    Main system class that orchestrates the entire RAG pipeline
    """

    def __init__(self):
        self.setup_graphiti()
        self.setup_neo4j_constraints()
        self.setup_retrievers()

    def setup_graphiti(self):
        """Initialize Graphiti with optimized settings for mathematical content"""

        # Configure LLM for entity extraction (use smaller model for efficiency)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

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
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
        )

    def setup_neo4j_constraints(self):
        """Define the new, multi-pass graph schema with constraints and indexes."""

        # First, drop the old constraints and indexes to avoid conflicts
        drop_commands = [
            "DROP CONSTRAINT fact_id IF EXISTS",
            "DROP CONSTRAINT definition_id IF EXISTS",
            "DROP CONSTRAINT proof_id IF EXISTS",
            "DROP INDEX fact_content_index IF EXISTS",
            "DROP INDEX definition_content_index IF EXISTS",
            "DROP INDEX proof_content_index IF EXISTS",
        ]

        # New schema constraints and indexes
        setup_commands = [
            # Node uniqueness constraints
            "CREATE CONSTRAINT term_id IF NOT EXISTS FOR (t:Term) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT symbol_id IF NOT EXISTS FOR (s:Symbol) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT statement_id IF NOT EXISTS FOR (st:Statement) REQUIRE st.id IS UNIQUE",
            "CREATE CONSTRAINT argument_id IF NOT EXISTS FOR (a:Argument) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (src:Source) REQUIRE src.id IS UNIQUE",
            # Full-text search indexes for powerful searching
            "CREATE FULLTEXT INDEX term_content_index IF NOT EXISTS FOR (t:Term) ON EACH [t.term, t.definition]",
            "CREATE FULLTEXT INDEX symbol_content_index IF NOT EXISTS FOR (s:Symbol) ON EACH [s.symbol, s.definition]",
            "CREATE FULLTEXT INDEX statement_content_index IF NOT EXISTS FOR (st:Statement) ON EACH [st.content, st.explanation]",
            "CREATE FULLTEXT INDEX argument_content_index IF NOT EXISTS FOR (a:Argument) ON EACH [a.content, a.explanation]",
            "CREATE FULLTEXT INDEX source_title_index IF NOT EXISTS FOR (src:Source) ON EACH [src.title, src.authors]",
        ]

        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
        )

        with driver.session() as session:
            print("   -> Dropping old schema constraints and indexes...")
            for query in drop_commands:
                try:
                    session.run(query)
                    print(f"✓ Executed: {query}")
                except Exception as e:
                    print(f"⚠ Failed to execute drop command: {query} - {e}")

            print("\n   -> Setting up new schema constraints and indexes...")
            for query in setup_commands:
                try:
                    session.run(query)
                    print(f"✓ Executed: {query}")
                except Exception as e:
                    print(f"⚠ Skipped (already exists or error): {query} - {e}")

        driver.close()

    def setup_retrievers(self):
        """Initialize different retrieval strategies"""

        # This will be implemented in Phase 2
        self.retrievers = {}

    async def build_indices_and_constraints(self):
        """Build necessary database indices and constraints"""

        await self.graphiti.build_indices_and_constraints()
        print("✓ Graphiti indices and constraints created")


# --- Prompts for Multi-Pass Ingestion ---

# Pass 1: Glossary Creation (Terms and Symbols)
TERM_EXTRACTION_PROMPT = """
You are a specialist in mathematical and logical notation. Your sole task is to scan the following text and extract formal definitions for terminology and symbols.

**Your Goal:** Identify and isolate definitions. Ignore everything else. Do not extract theorems, proofs, examples, or general discussion.

**Output Schema (Strict JSON format):**
Provide a JSON object with a single key, "glossary_items". This is a list of all the definitions you found.

- `type`: Must be either "Term" or "Symbol".
- `id`: A unique, machine-readable ID (e.g., "def-illative-combinatory-logic", "sym-turnstile").
- `label`: The exact term or symbol being defined (e.g., "Illative Combinatory Logic", "⊢").
- `definition`: The verbatim text of the definition.
- `context`: A brief explanation of the context in which the definition appears.

**Example:**
```json
{
  "glossary_items": [
    {
      "type": "Term",
      "id": "def-reduction",
      "label": "Reduction",
      "definition": "A binary relation on the set of terms, denoted by ->.",
      "context": "This is the foundational definition of reduction, appearing at the start of the section on term rewriting."
    },
    {
      "type": "Symbol",
      "id": "sym-lambda",
      "label": "λ",
      "definition": "The symbol used to denote lambda abstraction.",
      "context": "Introduced in the section on lambda calculus syntax."
    }
  ]
}
```

**Important:** If the text contains no formal definitions, return an empty list: `{ "glossary_items": [] }`.

**Text to Analyze:**
---
{text}
---
"""

# Pass 2: Statement and Argument Extraction
STATEMENT_EXTRACTION_PROMPT = """
You are an expert in mathematical logic. Your task is to analyze the provided text and extract the core logical STATEMENTS (theorems, lemmas, propositions) and ARGUMENTS (proofs).

**Your Goal:** Create structured, self-contained, and interconnected representations of the logical content.

**Context is CRITICAL:**
You will be provided with a "Contextual Glossary" of terms and symbols that have already been defined in this document. You MUST use this glossary to understand the text and to create self-contained statements.

**Chain of Thought:**
1.  **Identify Candidates:** Read the text and identify potential STATEMENTS and ARGUMENTS.
2.  **Contextualize & Refine:** For each candidate:
    *   Use the provided **Contextual Glossary** to understand the meaning of the terms and symbols.
    *   Rephrase the statement to be fully self-contained. For example, instead of "The theorem is proven by induction," write "The Church-Rosser Theorem is proven by induction on the structure of lambda terms."
    *   Identify all the `terms` and `symbols` from the glossary that are used in the statement or argument.
3.  **Generate Relationships:** Explicitly define the relationships between the items you've extracted.

**Output Schema (Strict JSON format):**
Provide a JSON object with a single key, "entities". This is a list of all the statements and arguments you found.

- `type`: Must be either "Statement" or "Argument".
- `id`: A unique, machine-readable ID (e.g., "stmt-church-rosser-theorem").
- `content`: The full, self-contained text of the statement or argument.
- `explanation`: A brief, plain-language explanation of the item's significance.
- `uses_terms`: A list of the `id`s of the Terms from the glossary that are used.
- `uses_symbols`: A list of the `id`s of the Symbols from the glossary that are used.
- `proves`: (For Arguments only) The `id` of the Statement that this argument proves.

**Example:**
```json
{
  "entities": [
    {
      "type": "Statement",
      "id": "stmt-church-rosser-theorem",
      "content": "The relation of one-step reduction in the lambda calculus is confluent.",
      "explanation": "This is the Church-Rosser theorem, a cornerstone of lambda calculus.",
      "uses_terms": ["def-reduction", "def-confluence"],
      "uses_symbols": ["sym-lambda"]
    },
    {
      "type": "Argument",
      "id": "arg-proof-of-church-rosser",
      "content": "[... verbatim, self-contained proof text ...]",
      "explanation": "The proof proceeds by induction on the structure of lambda terms.",
      "proves": "stmt-church-rosser-theorem",
      "uses_terms": ["def-reduction"],
      "uses_symbols": []
    }
  ]
}
```

**Important:** If the text contains no statements or arguments, return an empty list: `{ "entities": [] }`.

**Contextual Glossary:**
---
{glossary}
---

**Text to Analyze:**
---
{text}
---
"""


# 6. System initialization
async def initialize_system():
    """Initialize the complete system"""

    print("🚀 Initializing Combinatory Logic RAG System...")

    # Create system instance
    system = CombinatoryLogicRAGSystem()

    # Build indices and constraints

    print("✅ System initialized successfully!")
    return system


# Usage example:
if __name__ == "__main__":
    import asyncio

    # Run the initialization
    system = asyncio.run(initialize_system())
    print("System ready for document ingestion and querying!")
