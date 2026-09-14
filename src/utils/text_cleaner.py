"""
Text cleaning utilities for ML book PDF extraction.
Optimized for chunking strategy with math-aware preprocessing.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Comprehensive text cleaner for ML book PDFs.
    Addresses extraction issues: missing spaces, OCR artifacts, encoding problems.
    """

    def __init__(self):
        # Common encoding issues in ML books
        self.encoding_fixes = {
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            "Ã©": "é",
            "Ã¨": "è",
            "Ã«": "ë",
            "Ã¶": "ö",
            "Ã¼": "ü",
            "Ã¢": "â",
            "â€“": "–",
            "â€”": "—",
            "Ã˜": "Ø",
            "Ã¥": "å",
            "Ã¦": "æ",
            "Ã§": "ç",
            "Ã±": "ñ",
        }

        # Math symbols that might appear garbled
        self.math_symbol_fixes = {
            "âˆ†": "∆",  # Delta
            "âˆ‘": "∑",  # Sum
            "âˆ": "√",  # Square root
            "âˆž": "∞",  # Infinity
            "âˆ«": "∫",  # Integral
            "âˆ‚": "∂",  # Partial
            "âˆ‡": "∇",  # Nabla
            "âˆˆ": "∈",  # Element of
            "âˆ‰": "∉",  # Not element of
            "âŠ‚": "⊂",  # Subset
            "âŠƒ": "⊃",  # Superset
            "âˆ©": "∩",  # Intersection
            "âˆª": "∪",  # Union
            "âˆ§": "∧",  # Logical AND
            "âˆ¨": "∨",  # Logical OR
            "âˆ€": "∀",  # For all
            "âˆƒ": "∃",  # There exists
            "Â±": "±",  # Plus minus
            "Ã—": "×",  # Multiplication
            "Â·": "·",  # Dot
            "â‰¤": "≤",  # Less than or equal
            "â‰¥": "≥",  # Greater than or equal
            "â‰ ": "≠",  # Not equal
        }

        # Common abbreviations and their expansions
        self.abbreviations = {
            "e.g": "e.g.",
            "i.e": "i.e.",
            "et al": "et al.",
            "vs": "vs.",
            "Fig": "Figure",
            "Eq": "Equation",
            "Sect": "Section",
            "Chapt": "Chapter",
            "App": "Appendix",
        }

    def clean_text(
        self,
        text: str,
        fix_spaces: bool = True,
        fix_encoding: bool = True,
        fix_math: bool = True,
        normalize_whitespace: bool = True,
    ) -> str:
        """
        Main cleaning pipeline for extracted text.

        Args:
            text: Raw text from PDF extraction
            fix_spaces: Fix missing spaces (48% of pages affected)
            fix_encoding: Fix encoding issues
            fix_math: Fix math symbol encoding
            normalize_whitespace: Normalize whitespace and line breaks

        Returns:
            Cleaned text ready for chunking
        """
        if not text:
            return ""

        cleaned = text

        # Apply fixes in logical order
        if fix_spaces:
            cleaned = self.fix_missing_spaces(cleaned)

        if fix_encoding:
            cleaned = self.fix_encoding_issues(cleaned)

        if fix_math:
            cleaned = self.fix_math_symbols(cleaned)

        # Clean up specific ML book artifacts
        cleaned = self.clean_ml_artifacts(cleaned)

        # Fix common abbreviations
        cleaned = self.fix_abbreviations(cleaned)

        if normalize_whitespace:
            cleaned = self.normalize_whitespace(cleaned)

        return cleaned

    def fix_missing_spaces(self, text: str) -> str:
        """
        Fix missing spaces in PDF-extracted text.
        Addresses the 48% of pages with this issue in the quality report.

        Patterns fixed:
        - wordWord -> word Word (camelCase split)
        - word123 -> word 123 (word-number split)
        - 123word -> 123 word (number-word split)
        - word.word -> word. word (period followed by word)
        - word,word -> word, word (comma followed by word)
        - word;word -> word; word (semicolon followed by word)
        - word( -> word ( (word followed by parenthesis)
        - )word -> ) word (parenthesis followed by word)
        - word=word -> word = word (operators)
        - word<word -> word < word (inequalities)
        - word>word -> word > word (inequalities)
        """
        # Pattern 1: Word followed by capital letter (camelCase)
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        # Pattern 2: Word followed by number
        text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
        text = re.sub(r"([A-Z])(\d)", r"\1 \2", text)

        # Pattern 3: Number followed by word
        text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)

        # Pattern 4: Period followed by capital letter (but not abbreviation)
        text = re.sub(r"\.([A-Z][a-z])", r". \1", text)

        # Pattern 5: Comma or semicolon followed by word
        text = re.sub(r"([,;])([a-zA-Z])", r"\1 \2", text)

        # Pattern 6: Closing parenthesis followed by word
        text = re.sub(r"\)([a-zA-Z])", r") \1", text)

        # Pattern 7: Word followed by opening parenthesis
        text = re.sub(r"([a-zA-Z])(\()", r"\1 \2", text)

        # Pattern 8: Mathematical operators
        text = re.sub(r"([a-zA-Z])([=<>])", r"\1 \2", text)
        text = re.sub(r"([=<>])([a-zA-Z])", r"\1 \2", text)

        # Pattern 9: Colon followed by word
        text = re.sub(r":([a-zA-Z])", r": \1", text)

        # Pattern 10: Percentage sign followed by number
        text = re.sub(r"%(\d)", r"% \1", text)

        # Pattern 11: Number followed by percentage sign
        text = re.sub(r"(\d)%", r"\1 %", text)

        return text

    def fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues from PDF extraction"""
        for bad, good in self.encoding_fixes.items():
            text = text.replace(bad, good)
        return text

    def fix_math_symbols(self, text: str) -> str:
        """Fix math symbols that appear garbled"""
        for bad, good in self.math_symbol_fixes.items():
            text = text.replace(bad, good)
        return text

    def clean_ml_artifacts(self, text: str) -> str:
        """
        Clean ML-specific artifacts and formatting issues.
        """
        # Fix repeated characters (OCR artifacts)
        text = re.sub(r"([a-zA-Z])\1{3,}", r"\1\1", text)

        # Fix repeated punctuation
        text = re.sub(r"([.!?])\1{2,}", r"\1", text)

        # Fix common page break artifacts
        text = re.sub(r"-{2,}", "—", text)
        text = re.sub(r"_{2,}", "_", text)
        text = re.sub(r"\*{2,}", "*", text)

        # Fix LaTeX-like escape sequences
        text = re.sub(r"\\\(", " (", text)
        text = re.sub(r"\\\)", ") ", text)
        text = re.sub(r"\\\[", "[", text)
        text = re.sub(r"\\\]", "] ", text)

        # Fix common math notation issues
        text = re.sub(r"([a-zA-Z])\s*\^\s*([0-9])", r"\1^\2", text)  # x ^ 2 -> x^2
        text = re.sub(r"([a-zA-Z])\s*_\s*([0-9a-zA-Z])", r"\1_\2", text)  # x _ 2 -> x_2

        # Fix common references
        text = re.sub(
            r"(Chapter|Section|Equation|Figure|Table|Algorithm)\s+([0-9]+)\s+([0-9]+)",
            r"\1 \2.\3",
            text,
        )  # Section 5 3 -> Section 5.3

        return text

    def fix_abbreviations(self, text: str) -> str:
        """Fix common abbreviations in ML literature"""
        for short, full in self.abbreviations.items():
            text = text.replace(short, full)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace and line breaks for consistent chunking.
        """
        # Replace multiple newlines with double newline (section breaks)
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)

        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)

        # Remove spaces before punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)

        # Add space after punctuation if missing
        text = re.sub(r"([.,;:!?])([a-zA-Z])", r"\1 \2", text)

        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def preserve_math_blocks(self, text: str) -> str:
        """
        Identify and protect math blocks from being broken during cleaning.
        Useful for chunking strategy.
        """
        # Mark LaTeX math blocks that should be preserved
        math_patterns = [
            (r"\$[^\$]+\$", "inline_math"),  # Inline math
            (r"\\\[.*?\\\]", "display_math"),  # Display math
            (r"\\\(.*?\\\)", "inline_math"),  # Parenthesis math
            (r"\\begin\{.*?\}.*?\\end\{.*?\}", "env_math"),  # Math environments
        ]

        protected_blocks = []

        for pattern, math_type in math_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                block = match.group()
                placeholder = f"__{math_type.upper()}_{len(protected_blocks)}__"
                protected_blocks.append(
                    {"placeholder": placeholder, "content": block, "type": math_type}
                )
                text = text.replace(block, placeholder, 1)

        # Store protected blocks for later restoration
        self._math_blocks = protected_blocks
        return text

    def restore_math_blocks(self, text: str) -> str:
        """Restore protected math blocks after cleaning"""
        if hasattr(self, "_math_blocks"):
            for block_info in self._math_blocks:
                text = text.replace(block_info["placeholder"], block_info["content"])
        return text

    def extract_equations(self, text: str) -> list[dict[str, Any]]:
        """
        Extract equations for separate processing.
        Critical for math-aware chunking.
        """
        equations = []

        # Patterns to detect equations
        patterns = [
            (r"\$[^\$]+\$", "inline"),
            (r"\\\[.*?\\\]", "display"),
            (r"\\\(.*?\\\)", "inline"),
            (r"\\begin\{equation\}.*?\\end\{equation\}", "display"),
            (r"\\begin\{align\}.*?\\end\{align\}", "display"),
            (r"\\begin\{eqnarray\}.*?\\end\{eqnarray\}", "display"),
            (r"[=≠≤≥±×÷][\s]*[0-9a-zA-Z()\^_{}]+", "simple"),  # Simple equations
            (r"[∑∫∏√∂∇∞∈∉⊂⊃∩∪∧∨∀∃¬]", "symbol"),  # Math symbols
        ]

        for pattern, eq_type in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                equations.append(
                    {
                        "content": match.group(),
                        "type": eq_type,
                        "start": match.start(),
                        "end": match.end(),
                        "context": self._get_context(text, match.start(), match.end()),
                    }
                )

        return equations

    def _get_context(self, text: str, start: int, end: int, context_size: int = 100) -> str:
        """Get surrounding context for an equation"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end]

    def detect_section_boundaries(self, text: str) -> list[int]:
        """
        Detect section boundaries for chunking.
        Important for structural preservation.
        """
        boundaries = []
        lines = text.split("\n")
        pos = 0

        # Section patterns from quality report (916 sections, 1712 subsections)
        section_patterns = [
            r"^Chapter\s+\d+",
            r"^Section\s+\d+\.?\d*",
            r"^\d+\.\d+\s+[A-Z][a-z]",  # Subsection
            r"^\d+\.\d+\.\d+\s+[A-Z][a-z]",  # Subsubsection
            r"^(Abstract|Introduction|Conclusion|Appendix|References)",
            r"^[A-Z][A-Z\s]+$",  # ALL CAPS headers
        ]

        for _i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                pos += len(line) + 1
                continue

            for pattern in section_patterns:
                if re.match(pattern, line_stripped):
                    boundaries.append(pos)
                    break

            pos += len(line) + 1

        # Also add page boundaries (every ~500 words)
        words = text.split()
        for i, _word in enumerate(words):
            if i > 0 and i % 500 == 0:  # Every 500 words
                # Find the actual character position
                char_pos = 0
                for j in range(i):
                    char_pos += len(words[j]) + 1
                boundaries.append(char_pos)

        return sorted(set(boundaries))

    def get_chunking_stats(self, text: str) -> dict[str, Any]:
        """
        Get statistics useful for chunking decisions.
        Based on the quality report analysis.
        """
        words = text.split()
        characters = len(text)

        # Detect math density
        math_pattern = re.compile(r"[\$\\∑∫∏√∂∇∞∈∉⊂⊃∩∪∧∨∀∃¬=≠≤≥±×÷\^_{}]")
        math_chars = len(re.findall(math_pattern, text))

        # Detect equations
        eq_patterns = [r"\$[^\$]+\$", r"\\\[.*?\\\]", r"\\\(.*?\\\)"]
        equation_count = 0
        for pattern in eq_patterns:
            equation_count += len(re.findall(pattern, text, re.DOTALL))

        return {
            "word_count": len(words),
            "char_count": characters,
            "math_density": math_chars / characters if characters > 0 else 0,
            "equation_count": equation_count,
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "has_math": equation_count > 0 or math_chars > 0,
        }

    def clean_for_chunking(self, text: str, preserve_math: bool = True) -> str:
        """
        Specialized cleaning for chunking.
        Preserves math blocks and section structure.
        """
        # Step 1: Extract and protect math blocks
        math_blocks = self.extract_equations(text)
        if preserve_math and math_blocks:
            # Use placeholders for math blocks
            for i, block in enumerate(math_blocks):
                placeholder = f"__MATH_BLOCK_{i}__"
                text = text.replace(block["content"], placeholder)

        # Step 2: Clean the text
        cleaned = self.clean_text(text)

        # Step 3: Restore math blocks
        if preserve_math and math_blocks:
            for i, block in enumerate(math_blocks):
                placeholder = f"__MATH_BLOCK_{i}__"
                cleaned = cleaned.replace(placeholder, block["content"])

        return cleaned
