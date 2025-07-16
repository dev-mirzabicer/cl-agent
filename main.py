# A Simplified CLI for the Combinatory Logic Assistant

import asyncio
import logging

from infrastructure_setup import initialize_system
from multi_turn_agent import CombinatoryLogicAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CombinatoryLogicCLI:
    """
    A simple command-line interface to interact with the new agent.
    """

    def __init__(self):
        self.system = None
        self.agent = None

    async def initialize(self):
        """Initializes the system and the agent."""
        logger.info("🚀 Initializing Combinatory Logic Assistant...")
        self.system = await initialize_system()
        self.agent = CombinatoryLogicAgent(self.system)
        logger.info("✅ Assistant is ready.")

    async def run(self):
        """Runs the main CLI loop."""
        await self.initialize()

        print("\n🧠 Combinatory Logic Assistant CLI")
        print("=" * 40)
        print("Type 'quit' to exit.")

        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() in ["quit", "exit"]:
                    break
                if not query:
                    continue

                print("\nAssistant: ", end="", flush=True)
                response = await self.agent.query(query)
                print(response)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"An error occurred: {e}", exc_info=True)
                print(f"\n❌ Error: {e}")

        await self.close()
        print("\n👋 Goodbye!")


    async def close(self):
        """Closes the system resources."""
        if self.agent:
            await self.agent.close()


if __name__ == "__main__":
    cli = CombinatoryLogicCLI()
    asyncio.run(cli.run())
