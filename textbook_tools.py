# Tools for Interacting with the Textbook Project

import os
from typing import List, Dict, Any, Optional
import re


class TextbookManager:
    """
    A tool for the agent to read and modify the textbook project files.
    """

    def __init__(self, textbook_dir: str = "textbook"):
        self.textbook_dir = textbook_dir

    def _get_path(self, filename: str) -> str:
        return os.path.join(self.textbook_dir, filename)

    def read_file(self, filename: str) -> str:
        """Reads the content of one of the project files (PLAN.md, TASKS.md, etc.)."""
        try:
            with open(self._get_path(filename), "r") as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File '{filename}' not found."

    def write_file(self, filename: str, content: str) -> str:
        """Writes content to one of the project files."""
        with open(self._get_path(filename), "w") as f:
            f.write(content)
        return f"Successfully wrote to {filename}."

    def get_textbook_outline(self) -> str:
        """
        Parses the main .tex file and returns a high-level outline (table of contents).
        """
        tex_content = self.read_file("Textbook.tex")
        if tex_content.startswith("Error"):
            return tex_content

        outline = []
        for line in tex_content.split("\n"):
            if line.strip().startswith(r"\chapter{") or line.strip().startswith(
                r"\section{"
            ):
                # A simple regex to extract the title
                match = re.search(r"\{(.*?)\}", line)
                if match:
                    title = match.group(1)
                    if line.strip().startswith(r"\chapter{"):
                        outline.append(f"- Chapter: {title}")
                    else:
                        outline.append(f"  - Section: {title}")
            elif line.strip().startswith(r"\subsection{"):
                match = re.search(r"\{(.*?)\}", line)
                if match:
                    title = match.group(1)
                    outline.append(f"    - Subsection: {title}")

        if not outline:
            return "No outline found. The textbook may be empty."

        return "Textbook Outline:\n" + "\n".join(outline)

    def get_textbook_section_by_title(self, section_title: str) -> str:
        """
        Retrieves the full text of a specific chapter or section from the .tex file.
        """
        tex_content = self.read_file("Textbook.tex")
        if tex_content.startswith("Error"):
            return tex_content

        # This is a simplified implementation. A more robust version would
        # need to handle nested sections and more complex LaTeX structures.
        pattern = rf"\\(chapter|section|subsection)\{{ {re.escape(section_title)} \}}(.*?)(?=\\chapter|\\section|\\subsection|\\end\{{document\}})"
        match = re.search(pattern, tex_content, re.DOTALL)

        if match:
            return match.group(2).strip()
        else:
            return f"Error: Section '{section_title}' not found."
