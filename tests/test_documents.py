from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from demo_documents import render_documents, render_equation


class DocumentTests(unittest.TestCase):
    def test_equations_render_as_mathml_and_code_stays_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text(
                "# Method\n\n```math\nW' = W + \\alpha\\,g d^\\top\n```\n\n"
                "```math\nx^\\top W' = x^\\top W + \\alpha\\,(x^\\top g)\\,d^\\top\n```\n\n"
                "```python\nx @ W\n```\n")
            rendered = render_documents(root)["README.md"]["html"]
            parsed = ET.fromstring('<article>' + rendered + '</article>')
            equations = parsed.findall("div")
            self.assertEqual(len(equations), 2)
            for equation in equations:
                self.assertEqual(equation.get("class"), "doc-equation")
                math = equation.find("{http://www.w3.org/1998/Math/MathML}math")
                self.assertEqual(math.get("display"), "block")
                self.assertIn("α", ''.join(math.itertext()))
                self.assertIn("⊤", ''.join(math.itertext()))
            self.assertIn('<code class="language-python">x @ W', rendered)
            self.assertNotIn('language-math', rendered)

    def test_equations_reject_html_and_active_attributes(self):
        for source in (r'\text{<script>alert(1)</script>}',
                       r'\text{<img src="x" onerror="alert(1)"/>}',
                       r'\href{javascript:alert(1)}{x}'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                render_equation(source)

    def test_document_reader_formats_markdown_and_routes_nested_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experimental").mkdir()
            (root / ".pytest_cache").mkdir()
            (root / ".pytest_cache/README.md").write_text('# Cache\n\n[Not a document](missing.md)\n')
            (root / "README.md").write_text(
                '# Method\n\n[Patch](experimental/README.md#apply-it)\n\n'
                '| Part | Size |\n| --- | --- |\n| Adapter | 768 |\n\n'
                '```python\nprint("<script>")\n```\n\n'
                '## Setup\n\n## Setup\n\n<script>alert(1)</script>\n')
            (root / "experimental/README.md").write_text(
                '# Experimental\n\n## Apply it\n\n[Main](../README.md)\n\n'
                '[Here](#apply-it)\n\n[Patch](gatefix.safetensors)\n\n'
                '[External](https://example.com/source.md#code)\n')
            (root / "MODEL_CARD.md").write_text('---\nlicense: cc-by-4.0\n---\n\n# Model\n\n1. First\n2. Second\n')
            docs = render_documents(root)
            self.assertEqual(set(docs), {"README.md", "experimental/README.md", "MODEL_CARD.md"})
            main = docs["README.md"]["html"]
            self.assertEqual(docs["README.md"]["title"], "Method & setup")
            self.assertIn('href="#doc=experimental%2FREADME.md&amp;section=apply-it"', main)
            self.assertIn('<table>', main)
            self.assertIn('<pre tabindex="0"><code class="language-python">', main)
            self.assertIn('id="doc-setup"', main)
            self.assertIn('id="doc-setup-1"', main)
            self.assertNotIn('<script>', main)
            self.assertIn('&lt;script&gt;', main)
            nested = docs["experimental/README.md"]["html"]
            self.assertIn('href="#doc=README.md"', nested)
            self.assertIn('href="experimental/gatefix.safetensors"', nested)
            self.assertIn('href="https://example.com/source.md#code"', nested)
            self.assertNotIn('license:', docs["MODEL_CARD.md"]["html"])
            self.assertIn('<ol>', docs["MODEL_CARD.md"]["html"])
            hosted = render_documents(root, "/blog/piano-to-flute/")
            self.assertIn('href="/blog/piano-to-flute/experimental/gatefix.safetensors"', hosted["experimental/README.md"]["html"])
            self.assertIn('href="#doc=README.md"', hosted["experimental/README.md"]["html"])

    def test_document_reader_rejects_missing_local_markdown_and_unsafe_markup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "README.md"
            path.write_text('# Method\n\n[Missing](missing.md)\n')
            with self.assertRaisesRegex(ValueError, "Missing bundled document"):
                render_documents(root)
            path.write_text('# Method\n\n[Bad](javascript:alert%281%29)\n\n<img src=x onerror=alert(1)>\n')
            result = render_documents(root)["README.md"]["html"]
            self.assertNotIn('href="javascript:', result)
            self.assertNotIn('<img', result)
