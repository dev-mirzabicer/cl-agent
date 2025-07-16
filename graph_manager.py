import asyncio
from typing import List, Dict, Any, Optional

from neo4j import AsyncGraphDatabase
from neo4j.work.summary import ResultSummary


class GraphManager:
    """
    Manages all interactions with the Neo4j knowledge graph.
    """

    def __init__(self, system):
        self.system = system
        self.driver = system.graphiti.driver

    async def add_source(self, source_data: Dict[str, Any]) -> None:
        """Adds a new source node to the graph."""
        query = """
        MERGE (s:Source {id: $id})
        SET s.title = $title,
            s.authors = $authors,
            s.publication_year = $publication_year,
            s.source_path = $source_path
        """
        await self.driver.execute_query(
            query,
            id=source_data["id"],
            title=source_data["title"],
            authors=source_data["authors"],
            publication_year=source_data["publication_year"],
            source_path=source_data["source_path"],
        )

    async def add_term(self, term_data: Dict[str, Any]) -> None:
        """Adds a new Term node to the graph."""
        query = """
        MERGE (t:Term {id: $id})
        SET t.label = $label,
            t.definition = $definition,
            t.context = $context
        WITH t
        MATCH (s:Source {id: $source_id})
        MERGE (t)-[:DEFINED_IN]->(s)
        """
        await self.driver.execute_query(
            query,
            id=term_data["id"],
            label=term_data["label"],
            definition=term_data["definition"],
            context=term_data.get("context", ""),
            source_id=term_data["source_id"],
        )

    async def add_symbol(self, symbol_data: Dict[str, Any]) -> None:
        """Adds a new Symbol node to the graph."""
        query = """
        MERGE (s:Symbol {id: $id})
        SET s.label = $label,
            s.definition = $definition,
            s.context = $context
        WITH s
        MATCH (src:Source {id: $source_id})
        MERGE (s)-[:DEFINED_IN]->(src)
        """
        await self.driver.execute_query(
            query,
            id=symbol_data["id"],
            label=symbol_data["label"],
            definition=symbol_data["definition"],
            context=symbol_data.get("context", ""),
            source_id=symbol_data["source_id"],
        )

    async def add_statement(self, stmt_data: Dict[str, Any]) -> None:
        """Adds a new Statement node and its relationships to the graph."""
        print(f"         - GraphManager: Adding Statement '{stmt_data.get('id')}'")
        print(f"           - Data received: {list(stmt_data.keys())}")

        uses_terms = stmt_data.get("uses_terms", [])
        if not isinstance(uses_terms, list):
            print(
                f"           - WARNING: 'uses_terms' is not a list for statement {stmt_data.get('id')}. Skipping term relationships."
            )
            uses_terms = []

        uses_symbols = stmt_data.get("uses_symbols", [])
        if not isinstance(uses_symbols, list):
            print(
                f"           - WARNING: 'uses_symbols' is not a list for statement {stmt_data.get('id')}. Skipping symbol relationships."
            )
            uses_symbols = []

        query = """
        MERGE (st:Statement {id: $id})
        SET st.content = $content,
            st.explanation = $explanation
        WITH st
        MATCH (src:Source {id: $source_id})
        MERGE (st)-[:PART_OF]->(src)
        WITH st
        UNWIND $uses_terms AS term_id
        MATCH (t:Term {id: term_id})
        MERGE (st)-[:USES_TERM]->(t)
        WITH st
        UNWIND $uses_symbols AS symbol_id
        MATCH (s:Symbol {id: symbol_id})
        MERGE (st)-[:USES_SYMBOL]->(s)
        """
        await self.driver.execute_query(
            query,
            id=stmt_data["id"],
            content=stmt_data["content"],
            explanation=stmt_data.get("explanation", ""),
            source_id=stmt_data["source_id"],
            uses_terms=uses_terms,
            uses_symbols=uses_symbols,
        )

    async def add_argument(self, arg_data: Dict[str, Any]) -> None:
        """Adds a new Argument node and its relationships to the graph."""
        print(f"         - GraphManager: Adding Argument '{arg_data.get('id')}'")
        print(f"           - Data received: {list(arg_data.keys())}")

        proves = arg_data.get("proves")
        if not proves:
            print(
                f"           - WARNING: 'proves' field is missing for argument {arg_data.get('id')}. Skipping PROVES relationship."
            )
            return  # An argument must prove something

        uses_terms = arg_data.get("uses_terms", [])
        if not isinstance(uses_terms, list):
            print(
                f"           - WARNING: 'uses_terms' is not a list for argument {arg_data.get('id')}. Skipping term relationships."
            )
            uses_terms = []

        uses_symbols = arg_data.get("uses_symbols", [])
        if not isinstance(uses_symbols, list):
            print(
                f"           - WARNING: 'uses_symbols' is not a list for argument {arg_data.get('id')}. Skipping symbol relationships."
            )
            uses_symbols = []

        query = """
        MERGE (a:Argument {id: $id})
        SET a.content = $content,
            a.explanation = $explanation
        WITH a
        MATCH (src:Source {id: $source_id})
        MERGE (a)-[:PART_OF]->(src)
        WITH a
        MATCH (st:Statement {id: $proves})
        MERGE (a)-[:PROVES]->(st)
        WITH a
        UNWIND $uses_terms AS term_id
        MATCH (t:Term {id: term_id})
        MERGE (a)-[:USES_TERM]->(t)
        WITH a
        UNWIND $uses_symbols AS symbol_id
        MATCH (s:Symbol {id: symbol_id})
        MERGE (a)-[:USES_SYMBOL]->(s)
        """
        await self.driver.execute_query(
            query,
            id=arg_data["id"],
            content=arg_data["content"],
            explanation=arg_data.get("explanation", ""),
            source_id=arg_data["source_id"],
            proves=proves,
            uses_terms=uses_terms,
            uses_symbols=uses_symbols,
        )

    async def get_glossary_for_source(self, source_id: str) -> str:
        """Retrieves all terms and symbols for a given source to build a glossary."""
        query = """
        MATCH (t:Term)-[:DEFINED_IN]->(s:Source {id: $source_id})
        RETURN t.label as label, t.definition as definition, "Term" as type
        UNION ALL
        MATCH (sym:Symbol)-[:DEFINED_IN]->(s:Source {id: $source_id})
        RETURN sym.label as label, sym.definition as definition, "Symbol" as type
        """
        results = await self.driver.execute_query(query, source_id=source_id)

        glossary_parts = []
        if not results:
            return ""

        # The driver mixes records and a summary object in the results list.
        # We must filter out the summary before processing.
        records = [r for r in results[0]]

        # Access by index: 0=label, 1=definition, 2=type
        for record in records:
            if record and len(record.values()) == 3:
                label, definition, item_type = record.values()
                if label and definition:
                    glossary_parts.append(
                        f"- {item_type}: {label}\n  Definition: {definition}"
                    )

        return "\n".join(glossary_parts)

    async def get_global_glossary(self) -> str:
        """Retrieves all terms and symbols from the entire graph."""
        query = """
        MATCH (t:Term)
        RETURN t.label as label, t.definition as definition, "Term" as type
        UNION ALL
        MATCH (s:Symbol)
        RETURN s.label as label, s.definition as definition, "Symbol" as type
        """
        results, _, _ = await self.driver.execute_query(query)

        glossary_parts = []
        if not results:
            return ""

        for record in results:
            # Access by key
            label = record.get("label")
            definition = record.get("definition")
            item_type = record.get("type")
            if label and definition and item_type:
                glossary_parts.append(
                    f"- {item_type}: {label}\n  Definition: {definition}"
                )

        return "\n".join(glossary_parts)
