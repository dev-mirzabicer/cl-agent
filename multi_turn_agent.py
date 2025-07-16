# A Tool-Centric, Multi-Turn Agent for Combinatory Logic RAG

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from advanced_retrieval import KnowledgeGraphTool, DirectSourceReaderTool
from textbook_tools import TextbookManager


@dataclass
class AgentState:
    """A simple state for our tool-using agent."""

    messages: List[Any] = field(default_factory=list)


class CombinatoryLogicAgent:
    """
    A powerful, tool-centric agent that can reason about and use
    a suite of tools to answer complex queries about combinatory logic.
    """

    def __init__(self, system):
        self.system = system

        # Initialize the tools
        self.kg_tool = KnowledgeGraphTool(system)
        self.source_reader_tool = DirectSourceReaderTool(system)
        self.textbook_manager = TextbookManager()
        self.tools = {
            "search_knowledge_graph": self.kg_tool.search,
            "read_source": self.source_reader_tool.read_source,
            "read_textbook_file": self.textbook_manager.read_file,
            "write_textbook_file": self.textbook_manager.write_file,
            "get_textbook_outline": self.textbook_manager.get_textbook_outline,
            "get_textbook_section": self.textbook_manager.get_textbook_section,
        }

        # Initialize the powerful, tool-calling LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro-latest",
            temperature=0.1,
            max_tokens=4096,
            convert_system_message_to_human=True,
        )

        # Define the system prompt
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Builds the master system prompt for the agent."""
        return """
        You are a world-class expert in mathematical logic, specializing in combinatory logic, lambda calculus, and proof theory. Your purpose is to assist a user in writing a textbook on this subject. You have access to a suite of powerful tools to help you.

        **Your Capabilities:**
        1.  **Search the Knowledge Graph:** You can search a vast knowledge graph containing millions of facts, definitions, and proofs extracted from textbooks and research papers. Use the `search_knowledge_graph` tool for this. You can specify what you're looking for (e.g., nodes, edges) and how you want to search.
        2.  **Read Original Sources:** You can read the full text of any source document, chapter, or section in the knowledge base. Use the `read_source` tool for this. This is useful when you need to see the original context.
        3.  **Manage the Textbook Project:** You can read and write to the project's planning files (`PLAN.md`, `TASKS.md`, `NOTES.md`). You can also get an outline of the main `Textbook.tex` file and read specific sections from it. Use the `read_textbook_file`, `write_textbook_file`, `get_textbook_outline`, and `get_textbook_section` tools for this.

        **Your Workflow:**
        1.  **Analyze the Query:** Carefully analyze the user's query to understand their intent.
        2.  **Select a Tool:** Decide which tool is most appropriate for the query.
        3.  **Execute the Tool:** Call the selected tool with the correct parameters.
        4.  **Synthesize the Response:** Use the information returned by the tool to construct a comprehensive, accurate, and well-structured response.
        5.  **Multi-Turn Conversation:** Remember the context of the conversation. You can use information from previous turns to inform your tool usage and responses.

        Always provide mathematically rigorous and precise answers.
        """

    async def query(self, query: str, thread_id: str = "default") -> str:
        """
        Processes a query through the agent's reasoning loop.
        """
        print(f"🤖 Processing query: {query}")

        # For this simplified agent, we'll manage state in memory
        state = AgentState(messages=[HumanMessage(content=query)])

        # Add system prompt
        state.messages.insert(0, SystemMessage(content=self.system_prompt))

        # The agent's reasoning loop
        for _ in range(5):  # Limit to 5 turns to prevent infinite loops
            response = await self.llm.ainvoke(state.messages)
            state.messages.append(response)

            if not response.tool_calls:
                # If no tool is called, we're done
                return response.content

            # A tool was called, so execute it
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"🛠️ Calling tool: {tool_name} with args: {tool_args}")

                tool_function = self.tools.get(tool_name)
                if not tool_function:
                    tool_output = f"Error: Tool '{tool_name}' not found."
                else:
                    try:
                        tool_output = await tool_function(**tool_args)
                    except Exception as e:
                        tool_output = f"Error executing tool: {e}"

                state.messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                )

        # If we reach here, the loop finished, so return the last message
        return state.messages[-1].content

    async def close(self):
        """Close any resources used by the agent."""
        await self.kg_tool.close()
        await self.source_reader_tool.close()
