import json
import os
import re
import statistics
import sys
from collections import defaultdict
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExtractionQualityAnalyzer:
    """Comprehensive quality analysis for ML book PDF extraction"""

    def __init__(self, documents_path: str = "data/processed/chunks/documents_v1.json"):
        with open(documents_path, encoding="utf-8") as f:
            self.documents = json.load(f)

        # Patterns for detection
        self.math_patterns = {
            "latex_inline": r"\$[^\$]+\$",
            "latex_display": r"\\\[.*?\\\]",
            "latex_commands": r"\\(begin|end|frac|sqrt|sum|int|prod|lim|log|sin|cos|tan|alpha|beta|gamma|delta|epsilon|theta|lambda|sigma|omega|partial|nabla|times|cdot|rightarrow|left|right)",
            "subscript_superscript": r"[a-zA-Z0-9]\^\{[^\}]+\}|[a-zA-Z0-9]\_\{[^\}]+\}",
            "math_symbols": r"[∑∫∏√∂∇∞∈∉⊂⊃∩∪∧∨∀∃¬]",
            "equation_env": r"\\begin\{equation\}.*?\\end\{equation\}",
            "align_env": r"\\begin\{align\}.*?\\end\{align\}",
            "math_operators": r"[=≠≤≥±×÷]",
        }

        self.encoding_issues = {
            "mojibake": r"[\uFFFD\u0100-\u017F\u0400-\u04FF]",  # Common encoding errors
            "control_chars": r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        }

        self.structural_patterns = {
            "section": r"^(Chapter|Section|§)\s+\d+",
            "subsection": r"^\d+\.\d+\s+",
            "figure": r"(Figure|Fig\.)\s+\d+",
            "table": r"Table\s+\d+",
            "algorithm": r"(Algorithm|Alg\.)\s+\d+",
            "definition": r"(Definition|Def\.)\s+\d+",
            "theorem": r"(Theorem|Thm\.)\s+\d+",
            "proof": r"Proof\.",
            "example": r"(Example|Ex\.)\s+\d+",
            "remark": r"(Remark|Rem\.)\s+\d+",
        }

        self.book_patterns = {
            "Pattern Recognition and Machine Learning": {
                "expected_math_density": 0.4,  # 40% pages with math
                "expected_chapters": 14,
            },
            "Deep Learning": {
                "expected_math_density": 0.35,
                "expected_chapters": 20,
            },
        }

    def analyze_extraction_quality(self) -> dict[str, Any]:
        """Main analysis function with comprehensive metrics"""
        results = {
            "basic_stats": self._get_basic_stats(),
            "page_quality": self._analyze_page_quality(),
            "math_content": self._analyze_math_content(),
            "text_quality": self._analyze_text_quality(),
            "structural_quality": self._analyze_structure(),
            "per_book_analysis": self._analyze_per_book(),
            "chunking_recommendations": {},
            "quality_score": 0,
            "critical_issues": [],
            "warnings": [],
        }

        # Generate quality score and recommendations
        results["quality_score"] = self._calculate_quality_score(results)
        results["chunking_recommendations"] = self._generate_chunking_recommendations(results)

        return results

    def _get_basic_stats(self) -> dict:
        """Basic document statistics"""
        total_pages = len(self.documents)
        total_chars = sum(len(d["content"]) for d in self.documents)
        total_words = sum(len(d["content"].split()) for d in self.documents)

        return {
            "total_pages": total_pages,
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_page": total_chars / total_pages,
            "avg_words_per_page": total_words / total_pages,
            "page_length_std": statistics.stdev([len(d["content"]) for d in self.documents])
            if total_pages > 1
            else 0,
            "word_length_std": statistics.stdev([len(d["content"].split()) for d in self.documents])
            if total_pages > 1
            else 0,
        }

    def _analyze_page_quality(self) -> dict:
        """Detailed page quality analysis"""
        page_quality = {
            "empty_pages": [],
            "very_short_pages": [],  # < 50 words
            "short_pages": [],  # 50-200 words
            "normal_pages": [],  # 200-800 words
            "long_pages": [],  # > 800 words
            "page_length_distribution": {},
            "content_density": {},
        }

        for doc in self.documents:
            page_num = doc["page_number"]
            content = doc["content"]
            word_count = len(content.split())
            char_count = len(content)

            # Classify pages by length
            if char_count < 100:
                page_quality["empty_pages"].append(page_num)
            elif word_count < 50:
                page_quality["very_short_pages"].append(page_num)
            elif word_count < 200:
                page_quality["short_pages"].append(page_num)
            elif word_count < 800:
                page_quality["normal_pages"].append(page_num)
            else:
                page_quality["long_pages"].append(page_num)

            # Content density (chars per word average)
            density = char_count / word_count if word_count > 0 else 0
            page_quality["content_density"][page_num] = density

            # Distribution bins
            bin_key = f"{word_count // 100 * 100}-{(word_count // 100 + 1) * 100}"
            page_quality["page_length_distribution"][bin_key] = (
                page_quality["page_length_distribution"].get(bin_key, 0) + 1
            )

        return page_quality

    def _analyze_math_content(self) -> dict:
        """Comprehensive math content analysis"""
        math_analysis = {
            "pages_with_math": [],
            "math_density_by_page": {},
            "math_pattern_counts": defaultdict(int),
            "equation_extraction_quality": {
                "complete_equations": 0,
                "broken_equations": 0,
                "inline_vs_display": {"inline": 0, "display": 0},
                "complexity_distribution": defaultdict(int),
            },
            "math_preservation_quality": 0,  # Score 0-100
            "math_context_quality": {},  # How well equations are explained
        }

        for doc in self.documents:
            page_num = doc["page_number"]
            content = doc["content"]

            # Detect math on page
            has_math = False
            page_math_count = 0

            for pattern_name, pattern in self.math_patterns.items():
                matches = re.findall(pattern, content, re.DOTALL)
                if matches:
                    has_math = True
                    page_math_count += len(matches)
                    math_analysis["math_pattern_counts"][pattern_name] += len(matches)

                    # Track equation complexity
                    if pattern_name == "latex_commands":
                        for match in matches:
                            if any(cmd in match for cmd in ["frac", "sqrt", "sum", "int"]):
                                math_analysis["equation_extraction_quality"][
                                    "complexity_distribution"
                                ]["complex"] += 1
                            else:
                                math_analysis["equation_extraction_quality"][
                                    "complexity_distribution"
                                ]["simple"] += 1

            if has_math:
                math_analysis["pages_with_math"].append(page_num)
                math_analysis["math_density_by_page"][page_num] = page_math_count

        # Calculate math preservation quality
        total_math_pages = len(math_analysis["pages_with_math"])
        total_pages = len(self.documents)
        math_analysis["math_preservation_quality"] = int(
            min(100, (total_math_pages / total_pages) * 200)
        )
        for doc in self.documents:
            content = doc["content"]
            # Check for math that might be broken
            if re.search(r"\$[^\$]*\$[^\$]*\$", content):
                math_analysis["equation_extraction_quality"]["broken_equations"] += 1
            elif re.search(r"\\\[.*?\\\]", content) or re.search(r"\\\(.*?\\\)", content):
                math_analysis["equation_extraction_quality"]["complete_equations"] += 1

        return math_analysis

    def _analyze_text_quality(self) -> dict:
        """Detailed text quality analysis"""
        text_quality = {
            "encoding_issues": defaultdict(list),
            "garbled_pages": [],
            "non_ascii_distribution": {},
            "repeated_char_pages": [],
            "missing_space_pages": [],
            "ocr_artifacts": defaultdict(int),
            "line_break_issues": [],
            "font_consistency": {},
            "language_consistency": {},
        }

        for doc in self.documents:
            page_num = doc["page_number"]
            content = doc["content"]

            # Check encoding issues
            non_ascii_count = sum(1 for c in content if ord(c) > 127)
            non_ascii_ratio = non_ascii_count / len(content) if content else 0
            text_quality["non_ascii_distribution"][page_num] = non_ascii_ratio

            if non_ascii_ratio > 0.1:
                text_quality["garbled_pages"].append(page_num)

            # Check for specific encoding issues
            for issue_name, pattern in self.encoding_issues.items():
                if re.search(pattern, content):
                    text_quality["encoding_issues"][issue_name].append(page_num)

            # Check for repeated characters (OCR artifacts)
            repeated_chars = re.findall(r"([a-zA-Z])\1{2,}", content)
            if repeated_chars:
                text_quality["repeated_char_pages"].append(page_num)
                text_quality["ocr_artifacts"]["repeated_chars"] += len(repeated_chars)

            # Check for missing spaces (common PDF issue)
            if re.search(r"[a-z][A-Z]", content) or re.search(r"[a-z]\d", content):
                text_quality["missing_space_pages"].append(page_num)
                text_quality["ocr_artifacts"]["missing_spaces"] += 1

            # Check line break issues
            if re.search(r"\n\s*\n\s*\n", content):
                text_quality["line_break_issues"].append(page_num)

        return text_quality

    def _analyze_structure(self) -> dict:
        """Analyze document structure preservation"""
        structure = {
            "section_headers": [],
            "subsection_headers": [],
            "section_hierarchy": defaultdict(int),
            "figure_captions": [],
            "table_captions": [],
            "definitions_found": [],
            "theorems_found": [],
            "structural_elements_by_page": defaultdict(list),
            "missing_structural_elements": [],
            "page_breaks_preserved": [],
        }

        for doc in self.documents:
            page_num = doc["page_number"]
            content = doc["content"]

            # Find structural elements
            for element_name, pattern in self.structural_patterns.items():
                matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                if matches:
                    structure["structural_elements_by_page"][page_num].extend(matches)

                    if element_name == "section":
                        structure["section_headers"].extend(matches)
                    elif element_name == "subsection":
                        structure["subsection_headers"].extend(matches)
                    elif element_name == "figure":
                        structure["figure_captions"].extend(matches)
                    elif element_name == "table":
                        structure["table_captions"].extend(matches)
                    elif element_name in ["definition", "theorem"]:
                        structure["definitions_found"].extend(
                            matches
                        ) if element_name == "definition" else structure["theorems_found"].extend(
                            matches
                        )

            # Track section hierarchy depth
            if any(
                re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
                for pattern in [r"^Chapter\s+\d+", r"^Section\s+\d+"]
            ):
                structure["section_hierarchy"]["level_1"] += 1
            elif re.search(r"^\d+\.\d+\s+", content, re.MULTILINE):
                structure["section_hierarchy"]["level_2"] += 1
            elif re.search(r"^\d+\.\d+\.\d+\s+", content, re.MULTILINE):
                structure["section_hierarchy"]["level_3"] += 1

        return structure

    def _analyze_per_book(self) -> dict:
        """Per-book quality analysis"""
        per_book = {}

        for doc in self.documents:
            book = doc["book_title"]
            if book not in per_book:
                per_book[book] = {
                    "pages": [],
                    "total_words": 0,
                    "total_chars": 0,
                    "math_pages": [],
                    "encoding_issues": [],
                    "structural_elements": defaultdict(int),
                    "quality_metrics": {},
                }

            per_book[book]["pages"].append(doc["page_number"])
            per_book[book]["total_words"] += len(doc["content"].split())
            per_book[book]["total_chars"] += len(doc["content"])

            # Check for math on this page
            for pattern in self.math_patterns.values():
                if re.search(pattern, doc["content"], re.DOTALL):
                    per_book[book]["math_pages"].append(doc["page_number"])
                    break

        # Calculate per-book quality metrics
        for book, data in per_book.items():
            total_pages = len(data["pages"])
            math_pages = len(set(data["math_pages"]))

            data["quality_metrics"] = {
                "math_density": math_pages / total_pages if total_pages > 0 else 0,
                "math_coverage": math_pages,
                "avg_words_per_page": data["total_words"] / total_pages if total_pages > 0 else 0,
                "avg_chars_per_page": data["total_chars"] / total_pages if total_pages > 0 else 0,
                "expected_math_density": self.book_patterns.get(book, {}).get(
                    "expected_math_density", 0.3
                ),
            }

            # Grade the extraction
            expected_density = self.book_patterns.get(book, {}).get("expected_math_density", 0.3)
            actual_density = data["quality_metrics"]["math_density"]
            data["quality_metrics"]["math_extraction_grade"] = self._grade_extraction(
                actual_density / expected_density if expected_density > 0 else 0
            )

        return per_book

    def _grade_extraction(self, ratio: float) -> str:
        """Grade extraction quality based on math density ratio"""
        if ratio >= 0.9:
            return "A"
        elif ratio >= 0.75:
            return "B"
        elif ratio >= 0.5:
            return "C"
        elif ratio >= 0.25:
            return "D"
        else:
            return "F"

    def _calculate_quality_score(self, results: dict) -> int:
        """Calculate overall quality score (0-100)"""
        score = 100

        # Deduct for empty pages
        empty_pages = len(results["page_quality"]["empty_pages"])
        score -= min(20, empty_pages * 2)

        # Deduct for garbled text
        garbled = len(results["text_quality"]["garbled_pages"])
        score -= min(30, garbled * 3)

        # Deduct for poor math extraction
        math_quality = results["math_content"]["math_preservation_quality"]
        score -= max(0, (100 - math_quality) * 0.5)

        # Deduct for encoding issues
        encoding_issues = sum(
            len(pages) for pages in results["text_quality"]["encoding_issues"].values()
        )
        score -= min(20, encoding_issues * 2)

        # Deduct for missing structure
        if not results["structural_quality"]["section_headers"]:
            score -= 10

        # Bonus for good structure preservation
        if len(results["structural_quality"]["section_headers"]) > 20:
            score = min(100, score + 5)

        return max(0, min(100, score))

    def _generate_chunking_recommendations(self, results: dict) -> dict:
        """Generate specific chunking recommendations based on quality analysis"""
        recommendations: dict[str, Any] = {
            "chunk_size": 0,
            "overlap_size": 0,
            "strategies": [],
            "special_cases": [],
            "preprocessing_needed": [],
        }

        # Determine optimal chunk size based on page quality
        avg_words = results["basic_stats"]["avg_words_per_page"]
        if avg_words < 300:
            recommendations["chunk_size"] = 512
            recommendations["overlap_size"] = 80
        elif avg_words < 500:
            recommendations["chunk_size"] = 768
            recommendations["overlap_size"] = 120
        else:
            recommendations["chunk_size"] = 1024
            recommendations["overlap_size"] = 150

        # Math-heavy content needs special handling
        math_density = (
            len(results["math_content"]["pages_with_math"]) / results["basic_stats"]["total_pages"]
        )
        if math_density > 0.3:
            recommendations["strategies"].append("math_aware_chunking")
            recommendations["special_cases"].append("equations_should_never_be_split")
            recommendations["preprocessing_needed"].append("extract_latex_for_equations")

            # Smaller chunks for math content
            if recommendations["chunk_size"] > 512:
                recommendations["chunk_size"] = 512

        # Garbled text needs preprocessing
        if results["text_quality"]["garbled_pages"]:
            recommendations["preprocessing_needed"].append("fix_encoding_errors")
            recommendations["strategies"].append("fallback_to_ocr")

        # Missing structure needs semantic chunking
        if not results["structural_quality"]["section_headers"]:
            recommendations["strategies"].append("semantic_chunking")
            recommendations["preprocessing_needed"].append("add_section_boundaries")

        # Many short pages - need to combine
        if (
            len(results["page_quality"]["short_pages"])
            > results["basic_stats"]["total_pages"] * 0.3
        ):
            recommendations["strategies"].append("combine_short_pages")
            recommendations["overlap_size"] = max(50, recommendations["overlap_size"] + 50)

        # OCR artifacts detected
        if results["text_quality"]["ocr_artifacts"]:
            recommendations["preprocessing_needed"].append("clean_ocr_artifacts")
            recommendations["preprocessing_needed"].append("fix_missing_spaces")

        # Add specific strategies for ML books
        recommendations["strategies"].extend(
            [
                "preserve_reference_context",  # Citations and references are important
                "keep_tables_intact",  # Tables often contain critical data
                "maintain_code_blocks",  # ML books have algorithms
            ]
        )

        # Chunking parameters
        recommendations["chunking_params"] = {
            "chunk_size": recommendations["chunk_size"],
            "overlap": recommendations["overlap_size"],
            "min_chunk_size": 100,
            "max_chunk_size": recommendations["chunk_size"] * 1.5,
            "split_method": "semantic"
            if "semantic_chunking" in recommendations["strategies"]
            else "fixed",
            "preserve_equations": "math_aware_chunking" in recommendations["strategies"],
        }

        return recommendations

    def generate_report(self, results: dict | None = None) -> str:
        """Generate human-readable report"""
        if results is None:
            results = self.analyze_extraction_quality()

        report = []
        report.append("=" * 80)
        report.append("📊 COMPREHENSIVE EXTRACTION QUALITY REPORT")
        report.append("=" * 80)

        # Overall Quality Score
        report.append(f"\n🎯 OVERALL QUALITY SCORE: {results['quality_score']}/100")
        report.append(f"   Grade: {self._grade_extraction(results['quality_score'] / 100)}")

        # Critical Issues
        if results["quality_score"] < 70:
            report.append("\n⚠️  CRITICAL ISSUES FOUND:")
            if results["text_quality"]["garbled_pages"]:
                report.append(
                    f"   • {len(results['text_quality']['garbled_pages'])} pages have garbled text"
                )
            if len(results["page_quality"]["empty_pages"]) > 10:
                report.append(
                    f"   • {len(results['page_quality']['empty_pages'])} empty pages found"
                )
            if results["math_content"]["math_preservation_quality"] < 50:
                report.append(
                    f"   • Poor math extraction quality ({results['math_content']['math_preservation_quality']:.1f}%)"
                )

        # Basic Statistics
        report.append("\n📈 BASIC STATISTICS:")
        report.append(f"   Total Pages: {results['basic_stats']['total_pages']:,}")
        report.append(f"   Total Words: {results['basic_stats']['total_words']:,}")
        report.append(f"   Total Characters: {results['basic_stats']['total_chars']:,}")
        report.append(f"   Avg Words/Page: {results['basic_stats']['avg_words_per_page']:.1f}")
        report.append(f"   Avg Chars/Page: {results['basic_stats']['avg_chars_per_page']:.1f}")

        # Page Quality
        report.append("\n📄 PAGE QUALITY:")
        pq = results["page_quality"]
        report.append(f"   Empty Pages (<100 chars): {len(pq['empty_pages'])}")
        report.append(f"   Very Short Pages (<50 words): {len(pq['very_short_pages'])}")
        report.append(f"   Short Pages (50-200 words): {len(pq['short_pages'])}")
        report.append(f"   Normal Pages (200-800 words): {len(pq['normal_pages'])}")
        report.append(f"   Long Pages (>800 words): {len(pq['long_pages'])}")

        if pq["empty_pages"]:
            report.append(f"   ⚠️  Empty pages: {pq['empty_pages'][:10]}")

        # Math Content Quality
        report.append("\n📐 MATH CONTENT QUALITY:")
        mc = results["math_content"]
        report.append(
            f"   Pages with Math: {len(mc['pages_with_math'])} ({len(mc['pages_with_math']) / results['basic_stats']['total_pages'] * 100:.1f}%)"
        )
        report.append(f"   Math Preservation Quality: {mc['math_preservation_quality']:.1f}%")

        if mc["equation_extraction_quality"]["complete_equations"] > 0:
            report.append(
                f"   Complete Equations Detected: {mc['equation_extraction_quality']['complete_equations']}"
            )
        if mc["equation_extraction_quality"]["broken_equations"] > 0:
            report.append(
                f"   ⚠️  Broken Equations: {mc['equation_extraction_quality']['broken_equations']}"
            )

        # Text Quality
        report.append("\n🔤 TEXT QUALITY:")
        tq = results["text_quality"]
        report.append(f"   Pages with Encoding Issues: {len(set(tq['encoding_issues'].keys()))}")
        report.append(f"   Garbled Pages: {len(tq['garbled_pages'])}")
        report.append(f"   Pages with Repeated Characters: {len(tq['repeated_char_pages'])}")
        report.append(f"   Pages with Missing Spaces: {len(tq['missing_space_pages'])}")

        if tq["garbled_pages"]:
            report.append(f"   ⚠️  Garbled pages: {tq['garbled_pages'][:10]}")

        # Structural Quality
        report.append("\n📚 STRUCTURAL QUALITY:")
        sq = results["structural_quality"]
        report.append(f"   Section Headers Found: {len(sq['section_headers'])}")
        report.append(f"   Subsection Headers Found: {len(sq['subsection_headers'])}")
        report.append(f"   Figure Captions Found: {len(sq['figure_captions'])}")
        report.append(f"   Table Captions Found: {len(sq['table_captions'])}")
        report.append(f"   Definitions Found: {len(sq['definitions_found'])}")
        report.append(f"   Theorems Found: {len(sq['theorems_found'])}")

        # Per-Book Analysis
        report.append("\n📖 PER-BOOK QUALITY:")
        for book, data in results["per_book_analysis"].items():
            report.append(f"\n   {book}:")
            report.append(f"      Pages: {len(data['pages'])}")
            report.append(f"      Words: {data['total_words']:,}")
            report.append(
                f"      Math Pages: {len(set(data['math_pages']))} ({data['quality_metrics']['math_density'] * 100:.1f}%)"
            )
            report.append(
                f"      Expected Math Density: {data['quality_metrics'].get('expected_math_density', 0.3) * 100:.1f}%"
            )
            report.append(
                f"      Extraction Grade: {data['quality_metrics'].get('math_extraction_grade', 'N/A')}"
            )
            report.append(
                f"      Avg Words/Page: {data['quality_metrics']['avg_words_per_page']:.1f}"
            )

        # Chunking Recommendations
        report.append("\n🔧 CHUNKING RECOMMENDATIONS:")
        rec = results["chunking_recommendations"]
        report.append(f"\n   Optimal Chunk Size: {rec['chunk_size']} tokens")
        report.append(f"   Optimal Overlap: {rec['overlap_size']} tokens")

        report.append("\n   Recommended Strategies:")
        for strategy in rec["strategies"]:
            report.append(f"      • {strategy.replace('_', ' ').title()}")

        if rec["preprocessing_needed"]:
            report.append("\n   Preprocessing Required:")
            for preprocess in rec["preprocessing_needed"]:
                report.append(f"      • {preprocess.replace('_', ' ').title()}")

        if rec["special_cases"]:
            report.append("\n   Special Cases to Handle:")
            for special in rec["special_cases"]:
                report.append(f"      • {special.replace('_', ' ').title()}")

        # Chunking Parameters
        report.append("\n   Recommended Chunking Parameters:")
        for key, value in rec["chunking_params"].items():
            report.append(f"      • {key.replace('_', ' ').title()}: {value}")

        # Action Items
        report.append("\n📋 ACTION ITEMS:")
        if results["quality_score"] < 80:
            report.append("   1. Consider re-extracting PDFs with better tools:")
            report.append("      - Use pymupdf4llm for structure preservation")
            report.append("      - Use Marker or Nougat for math-heavy content")
            report.append("      - Consider using pdfplumber as fallback")

        if results["math_content"]["math_preservation_quality"] < 60:
            report.append("   2. Improve math extraction:")
            report.append("      - Use Mathpix API or grobid for equations")
            report.append("      - Convert math to LaTeX before chunking")
            report.append("      - Store equations separately with context")

        if results["text_quality"]["garbled_pages"]:
            report.append("   3. Clean garbled text:")
            report.append("      - Fix encoding (UTF-8 normalization)")
            report.append("      - Use OCR for heavily corrupted pages")

        report.append("\n" + "=" * 80)
        report.append("✅ Analysis Complete! Use these insights to optimize chunking.")
        report.append("=" * 80)

        return "\n".join(report)

    def export_detailed_metrics(self, results: dict | None = None) -> dict:
        """Export detailed metrics for programmatic use"""
        if results is None:
            results = self.analyze_extraction_quality()

        # Flatten results for CSV export
        flat_metrics = {
            "quality_score": results["quality_score"],
            "total_pages": results["basic_stats"]["total_pages"],
            "total_words": results["basic_stats"]["total_words"],
            "avg_words_per_page": results["basic_stats"]["avg_words_per_page"],
            "math_density": len(results["math_content"]["pages_with_math"])
            / results["basic_stats"]["total_pages"],
            "math_preservation_quality": results["math_content"]["math_preservation_quality"],
            "empty_pages": len(results["page_quality"]["empty_pages"]),
            "garbled_pages": len(results["text_quality"]["garbled_pages"]),
            "sections_found": len(results["structural_quality"]["section_headers"]),
            "equations_found": results["math_content"]["equation_extraction_quality"][
                "complete_equations"
            ],
            "broken_equations": results["math_content"]["equation_extraction_quality"][
                "broken_equations"
            ],
            "recommended_chunk_size": results["chunking_recommendations"]["chunk_size"],
            "recommended_overlap": results["chunking_recommendations"]["overlap_size"],
        }

        return flat_metrics


def main():
    """Main execution function"""
    analyzer = ExtractionQualityAnalyzer()
    results = analyzer.analyze_extraction_quality()
    report = analyzer.generate_report(results)
    print(report)

    # Export metrics for further analysis
    metrics = analyzer.export_detailed_metrics(results)
    print("\n📊 Detailed metrics exported:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")

    # Save report to file
    with open("data/processed/chunks/extraction_quality_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Save metrics to JSON
    with open("data/processed/chunks/quality_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Report saved to: data/processed/chunks/extraction_quality_report.txt")
    print("✅ Metrics saved to: data/processed/chunks/quality_metrics.json")


if __name__ == "__main__":
    main()
