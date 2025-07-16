# textbook_tools.py
"""
Enhanced tools for interacting with the textbook project.
Provides comprehensive textbook management capabilities.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TextbookManager:
    """
    Enhanced tool for managing the combinatory logic textbook project.
    Provides sophisticated file management and LaTeX document parsing.
    """

    def __init__(self, textbook_dir: str = "textbook"):
        """
        Initialize the textbook manager.

        Args:
            textbook_dir: Directory containing the textbook project
        """
        self.textbook_dir = Path(textbook_dir)
        self.textbook_dir.mkdir(exist_ok=True)

        # Initialize project structure
        self._ensure_project_structure()

    def _ensure_project_structure(self):
        """Ensure the basic project structure exists."""
        required_files = {
            "PLAN.md": self._get_default_plan(),
            "TASKS.md": self._get_default_tasks(),
            "NOTES.md": self._get_default_notes(),
            "Textbook.tex": self._get_default_textbook_tex(),
        }

        for filename, default_content in required_files.items():
            file_path = self.textbook_dir / filename
            if not file_path.exists():
                logger.info(f"📝 Creating default {filename}")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(default_content)

    def _get_default_plan(self) -> str:
        """Get default content for PLAN.md."""
        return """# Combinatory Logic Textbook Plan

## Overall Structure

### Part I: Foundations
1. **Introduction to Combinatory Logic**
   - Historical context
   - Motivation and applications
   - Relationship to lambda calculus

2. **Basic Combinators**
   - S, K, I combinators
   - Composition and application
   - Normal forms

3. **Combinatorial Completeness**
   - Translation from lambda calculus
   - Proof of completeness
   - Examples and exercises

### Part II: Advanced Topics
4. **Typed Combinatory Logic**
   - Simple types
   - Polymorphism
   - Type inference

5. **Applications and Extensions**
   - Logic programming
   - Category theory connections
   - Modern developments

## Current Status
- [ ] Chapter 1: In progress
- [ ] Chapter 2: Planned
- [ ] Chapter 3: Planned

## Notes
- Focus on pedagogical clarity
- Include many worked examples
- Provide exercises with solutions
"""

    def _get_default_tasks(self) -> str:
        """Get default content for TASKS.md."""
        return """# Active Tasks

## High Priority
- [ ] Complete introduction chapter
- [ ] Add formal definitions for basic combinators
- [ ] Prove Church-Rosser theorem for combinatory logic

## Medium Priority  
- [ ] Add historical background section
- [ ] Create exercise sets for each chapter
- [ ] Develop notation reference

## Low Priority
- [ ] Add bibliographic references
- [ ] Create index of terms
- [ ] Review for mathematical accuracy

## Completed
- [x] Set up project structure
- [x] Created basic outline

## Notes
Remember to validate all mathematical content against the knowledge base.
"""

    def _get_default_notes(self) -> str:
        """Get default content for NOTES.md."""
        return """# Textbook Writing Notes

## Key Principles
- Mathematical rigor without sacrificing accessibility
- Build intuition before formal definitions
- Use consistent notation throughout
- Include motivating examples

## Notation Decisions
- Use λ for lambda abstraction
- Use → for function types
- Use ≡ for definitional equality

## Important Sources to Reference
- Hindley & Seldin: "Lambda-Calculus and Combinators"
- Curry & Feys: "Combinatory Logic"
- Barendregt: "The Lambda Calculus"

## Review Checklist
- [ ] All definitions are precise
- [ ] Examples are worked through completely
- [ ] Proofs are rigorous but accessible
- [ ] Cross-references are accurate
"""

    def _get_default_textbook_tex(self) -> str:
        """Get default content for Textbook.tex."""
        return r"""
\documentclass[11pt]{book}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{unicode-math}
\usepackage{xparse}

\title{An Introduction to Combinatory Logic}
\author{Generated with Knowledge Graph Assistant}
\date{\today}

% Theorem environments
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}

\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{note}[theorem]{Note}

\begin{document}

\maketitle
\tableofcontents

\chapter{Introduction to Combinatory Logic}

Combinatory logic is a formal system that provides an alternative foundation 
to lambda calculus for studying computation and mathematical logic.

\section{Basic Concepts}

\begin{definition}[Combinator]
A combinator is a function with no free variables.
\end{definition}

\section{The SKI System}

The SKI combinator calculus consists of three basic combinators:

\begin{itemize}
\item $I$ - the identity combinator
\item $K$ - the constant combinator  
\item $S$ - the substitution combinator
\end{itemize}

% More content will be added here...

\end{document}
"""

    def read_file(self, filename: str) -> str:
        """
        Read the content of a project file.

        Args:
            filename: Name of the file to read

        Returns:
            File content or error message
        """
        try:
            file_path = self.textbook_dir / filename
            if not file_path.exists():
                return f"Error: File '{filename}' not found in {self.textbook_dir}"

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info(f"📖 Read file: {filename} ({len(content)} characters)")
            return content

        except Exception as e:
            error_msg = f"Error reading file '{filename}': {e}"
            logger.error(error_msg)
            return error_msg

    def write_file(self, filename: str, content: str) -> str:
        """
        Write content to a project file.

        Args:
            filename: Name of the file to write
            content: Content to write

        Returns:
            Success message or error
        """
        try:
            file_path = self.textbook_dir / filename

            # Create backup if file exists
            if file_path.exists():
                backup_path = file_path.with_suffix(
                    f".bak.{int(datetime.now().timestamp())}"
                )
                file_path.rename(backup_path)
                logger.info(f"📋 Created backup: {backup_path.name}")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(
                f"✅ Successfully wrote to {filename} ({len(content)} characters)"
            )
            return f"Successfully wrote to {filename}."

        except Exception as e:
            error_msg = f"Error writing to file '{filename}': {e}"
            logger.error(error_msg)
            return error_msg

    def append_to_file(self, filename: str, content: str) -> str:
        """
        Append content to a project file.

        Args:
            filename: Name of the file to append to
            content: Content to append

        Returns:
            Success message or error
        """
        try:
            file_path = self.textbook_dir / filename

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(content)

            logger.info(
                f"✅ Successfully appended to {filename} ({len(content)} characters)"
            )
            return f"Successfully appended to {filename}."

        except Exception as e:
            error_msg = f"Error appending to file '{filename}': {e}"
            logger.error(error_msg)
            return error_msg

    def get_textbook_outline(self) -> str:
        """
        Parse the main LaTeX file and return a structured outline.

        Returns:
            Hierarchical outline of the textbook
        """
        try:
            tex_content = self.read_file("Textbook.tex")
            if tex_content.startswith("Error"):
                return tex_content

            outline = []
            current_chapter = None
            current_section = None

            # Enhanced regex patterns for different LaTeX commands
            patterns = {
                "part": re.compile(r"\\part\*?\{([^}]+)\}"),
                "chapter": re.compile(r"\\chapter\*?\{([^}]+)\}"),
                "section": re.compile(r"\\section\*?\{([^}]+)\}"),
                "subsection": re.compile(r"\\subsection\*?\{([^}]+)\}"),
                "subsubsection": re.compile(r"\\subsubsection\*?\{([^}]+)\}"),
                "theorem": re.compile(r"\\begin\{theorem\}(?:\[([^\]]+)\])?"),
                "definition": re.compile(r"\\begin\{definition\}(?:\[([^\]]+)\])?"),
                "lemma": re.compile(r"\\begin\{lemma\}(?:\[([^\]]+)\])?"),
            }

            lines = tex_content.split("\n")

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith("%"):
                    continue

                # Check for structural elements
                for element_type, pattern in patterns.items():
                    match = pattern.search(line)
                    if match:
                        title = (
                            match.group(1)
                            if match.group(1)
                            else f"Unnamed {element_type}"
                        )

                        if element_type == "part":
                            outline.append(f"PART: {title}")
                        elif element_type == "chapter":
                            current_chapter = len(outline)
                            outline.append(f"Chapter: {title}")
                            current_section = None
                        elif element_type == "section":
                            if current_chapter is not None:
                                outline.append(f"  Section: {title}")
                                current_section = len(outline) - 1
                            else:
                                outline.append(f"Section: {title}")
                        elif element_type == "subsection":
                            outline.append(f"    Subsection: {title}")
                        elif element_type == "subsubsection":
                            outline.append(f"      Subsubsection: {title}")
                        elif element_type in ["theorem", "definition", "lemma"]:
                            indent = "      " if current_section else "    "
                            outline.append(f"{indent}{element_type.title()}: {title}")

            if not outline:
                return "No structured content found in Textbook.tex"

            result = "📚 Textbook Outline:\n" + "\n".join(outline)
            logger.info(f"📋 Generated outline with {len(outline)} items")
            return result

        except Exception as e:
            error_msg = f"Error generating outline: {e}"
            logger.error(error_msg)
            return error_msg

    def get_textbook_section_by_title(self, section_title: str) -> str:
        """
        Retrieve the content of a specific section by its title.

        Args:
            section_title: Title of the section to retrieve

        Returns:
            Section content or error message
        """
        try:
            tex_content = self.read_file("Textbook.tex")
            if tex_content.startswith("Error"):
                return tex_content

            # Enhanced pattern matching for sections
            section_patterns = [
                rf"\\chapter\*?\{{\s*{re.escape(section_title)}\s*\}}",
                rf"\\section\*?\{{\s*{re.escape(section_title)}\s*\}}",
                rf"\\subsection\*?\{{\s*{re.escape(section_title)}\s*\}}",
                rf"\\subsubsection\*?\{{\s*{re.escape(section_title)}\s*\}}",
            ]

            for pattern in section_patterns:
                match = re.search(pattern, tex_content, re.IGNORECASE)
                if match:
                    start_pos = match.end()

                    # Find the end of this section (next section at same or higher level)
                    remaining_content = tex_content[start_pos:]

                    # Define hierarchy for determining section end
                    section_level = None
                    if "\\chapter" in match.group(0):
                        section_level = 1
                        end_patterns = [r"\\chapter", r"\\end\{document\}"]
                    elif "\\section" in match.group(0):
                        section_level = 2
                        end_patterns = [
                            r"\\chapter",
                            r"\\section",
                            r"\\end\{document\}",
                        ]
                    elif "\\subsection" in match.group(0):
                        section_level = 3
                        end_patterns = [
                            r"\\chapter",
                            r"\\section",
                            r"\\subsection",
                            r"\\end\{document\}",
                        ]
                    else:  # subsubsection
                        section_level = 4
                        end_patterns = [
                            r"\\chapter",
                            r"\\section",
                            r"\\subsection",
                            r"\\subsubsection",
                            r"\\end\{document\}",
                        ]

                    # Find the next section at same or higher level
                    end_match = None
                    for end_pattern in end_patterns:
                        end_match = re.search(end_pattern, remaining_content)
                        if end_match:
                            break

                    if end_match:
                        section_content = remaining_content[: end_match.start()].strip()
                    else:
                        section_content = remaining_content.strip()

                    if section_content:
                        result = f"📖 Section: {section_title}\n{'=' * 50}\n{section_content}"
                        logger.info(
                            f"📋 Retrieved section '{section_title}' ({len(section_content)} characters)"
                        )
                        return result
                    else:
                        return (
                            f"Section '{section_title}' found but appears to be empty."
                        )

            return f"Error: Section '{section_title}' not found in textbook."

        except Exception as e:
            error_msg = f"Error retrieving section '{section_title}': {e}"
            logger.error(error_msg)
            return error_msg

    def get_textbook_section(self, section_title: str) -> str:
        """Alias for get_textbook_section_by_title for backward compatibility."""
        return self.get_textbook_section_by_title(section_title)

    def list_project_files(self) -> List[str]:
        """
        List all files in the textbook project directory.

        Returns:
            List of file names
        """
        try:
            files = []
            for file_path in self.textbook_dir.iterdir():
                if file_path.is_file():
                    files.append(file_path.name)

            logger.info(f"📁 Found {len(files)} files in project directory")
            return sorted(files)

        except Exception as e:
            logger.error(f"Error listing project files: {e}")
            return []

    def search_content(
        self, query: str, file_pattern: str = "*.md"
    ) -> List[Dict[str, Any]]:
        """
        Search for content across project files.

        Args:
            query: Search query
            file_pattern: File pattern to search (e.g., "*.md", "*.tex")

        Returns:
            List of search results with file names and line numbers
        """
        try:
            results = []
            search_files = list(self.textbook_dir.glob(file_pattern))

            for file_path in search_files:
                if file_path.is_file():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()

                        for line_num, line in enumerate(lines, 1):
                            if query.lower() in line.lower():
                                results.append(
                                    {
                                        "file": file_path.name,
                                        "line_number": line_num,
                                        "content": line.strip(),
                                        "context": "..." + line.strip() + "...",
                                    }
                                )
                    except Exception as e:
                        logger.warning(f"Could not search file {file_path}: {e}")

            logger.info(f"🔍 Found {len(results)} matches for '{query}'")
            return results

        except Exception as e:
            logger.error(f"Error searching content: {e}")
            return []

    def get_project_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the textbook project.

        Returns:
            Dictionary with project statistics
        """
        try:
            stats = {
                "files": {},
                "total_characters": 0,
                "total_lines": 0,
                "total_words": 0,
            }

            for file_path in self.textbook_dir.iterdir():
                if file_path.is_file() and file_path.suffix in [".md", ".tex", ".txt"]:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        lines = content.split("\n")
                        words = content.split()

                        file_stats = {
                            "characters": len(content),
                            "lines": len(lines),
                            "words": len(words),
                        }

                        stats["files"][file_path.name] = file_stats
                        stats["total_characters"] += file_stats["characters"]
                        stats["total_lines"] += file_stats["lines"]
                        stats["total_words"] += file_stats["words"]

                    except Exception as e:
                        logger.warning(f"Could not analyze file {file_path}: {e}")

            return stats

        except Exception as e:
            logger.error(f"Error getting project statistics: {e}")
            return {"error": str(e)}

    def export_project_summary(self) -> str:
        """
        Export a comprehensive summary of the textbook project.

        Returns:
            Formatted project summary
        """
        try:
            summary_parts = []

            # Project overview
            summary_parts.append("# Textbook Project Summary")
            summary_parts.append(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            summary_parts.append("")

            # Statistics
            stats = self.get_project_statistics()
            if "error" not in stats:
                summary_parts.append("## Project Statistics")
                summary_parts.append(f"- Total files: {len(stats['files'])}")
                summary_parts.append(
                    f"- Total characters: {stats['total_characters']:,}"
                )
                summary_parts.append(f"- Total lines: {stats['total_lines']:,}")
                summary_parts.append(f"- Total words: {stats['total_words']:,}")
                summary_parts.append("")

            # Outline
            outline = self.get_textbook_outline()
            if not outline.startswith("Error"):
                summary_parts.append("## Current Structure")
                summary_parts.append(outline)
                summary_parts.append("")

            # Recent tasks (from TASKS.md)
            tasks_content = self.read_file("TASKS.md")
            if not tasks_content.startswith("Error"):
                summary_parts.append("## Current Tasks")
                summary_parts.append(
                    tasks_content[:500] + "..."
                    if len(tasks_content) > 500
                    else tasks_content
                )
                summary_parts.append("")

            return "\n".join(summary_parts)

        except Exception as e:
            return f"Error generating project summary: {e}"
