"""Render bundled Markdown once, for the demo's offline document reader."""

import posixpath
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote, unquote, urlencode, urlsplit, urlunsplit

from markdown_it import MarkdownIt
from latex2mathml.converter import convert


DOCUMENT_TITLES = {
    "README.md": "Project overview",
    "docs/REPRODUCE.md": "Reproduce the edit",
    "docs/ATTRIBUTION.md": "Credits & licenses",
    "experimental/README.md": "Experimental patch",
    "docs/MODEL_CARD.md": "Model card",
}


def render_equation(source):
    """Typeset math at build time; accept presentation markup, never links or HTML."""
    markup = convert(source, display="block")
    namespace = "{http://www.w3.org/1998/Math/MathML}"
    tags = {namespace + name for name in (
        "math", "mrow", "mi", "mn", "mo", "mtext", "mspace", "msup", "msub",
        "msubsup", "mfrac", "msqrt", "mroot", "mover", "munder", "munderover",
        "mtable", "mtr", "mtd", "mstyle", "mpadded", "mphantom",
    )}
    attributes = {"display", "mathvariant", "stretchy", "width", "height", "depth",
                  "accent", "accentunder", "linethickness", "columnalign", "rowalign",
                  "columnspacing", "rowspacing", "displaystyle", "scriptlevel",
                  "fence", "separator", "lspace", "rspace", "minsize", "maxsize"}
    try:
        root = ET.fromstring(markup)
    except ET.ParseError as error:
        raise ValueError("Invalid markup in equation") from error
    for element in root.iter():
        if element.tag not in tags or not element.attrib.keys() <= attributes:
            raise ValueError("Unsupported markup in equation")
    return f'<div class="doc-equation" tabindex="0" role="region" aria-label="Equation">{markup}</div>\n'


def render_documents(directory, asset_base=""):
    sources = {path.relative_to(directory).as_posix(): path.read_text()
               for path in sorted(directory.rglob("*.md"))
               if not any(part.startswith(".") for part in path.relative_to(directory).parts)}
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    default_fence = parser.renderer.rules["fence"]

    def render_fence(tokens, index, options, env):
        if tokens[index].info.strip() == "math":
            return render_equation(tokens[index].content)
        return default_fence(tokens, index, options, env)

    parser.renderer.rules["fence"] = render_fence
    documents = {}
    for name in dict.fromkeys([*DOCUMENT_TITLES, *sources]):
        if name not in sources:
            continue
        source = sources[name]
        # Model-card metadata stays in the original file, outside the reading view.
        source = re.sub(r"\A---\r?\n.*?\r?\n---(?:\r?\n|$)", "", source, count=1, flags=re.S)
        tokens = parser.parse(source)
        headings = set()
        title = name
        for index, token in enumerate(tokens):
            if token.type == "heading_open":
                text = "".join(child.content for child in tokens[index + 1].children
                               if child.type in ("text", "code_inline"))
                if token.tag == "h1" and title == name:
                    title = text
                stem = re.sub(r"\s+", "-", re.sub(r"[^\w\-\s]", "", text.lower()))
                anchor, suffix = stem, 0
                while anchor in headings:
                    suffix += 1
                    anchor = f"{stem}-{suffix}"
                headings.add(anchor)
                token.attrSet("id", f"doc-{anchor}")
            for child in token.children or []:
                if child.type != "link_open":
                    continue
                url = urlsplit(child.attrGet("href"))
                if url.scheme or url.netloc:
                    continue
                target = (posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(url.path)))
                          if url.path else name)
                if target in sources:
                    route = {"doc": target}
                    if url.fragment:
                        route["section"] = unquote(url.fragment)
                    child.attrSet("href", "#" + urlencode(route))
                elif target.lower().endswith(".md"):
                    raise ValueError(f"Missing bundled document: {name} links to {target}")
                elif target == "index.html" and not url.query:
                    child.attrSet("href", "#" + (url.fragment or "top"))
                else:
                    child.attrSet("href", urlunsplit(("", "", asset_base + quote(target, safe="/"), url.query, url.fragment)))
        rendered = parser.renderer.render(tokens, parser.options, {})
        rendered = rendered.replace("<table>", '<div class="doc-table" tabindex="0" role="region" aria-label="Scrollable table"><table>')
        rendered = rendered.replace("</table>", "</table></div>")
        rendered = rendered.replace("<pre>", '<pre tabindex="0">')
        documents[name] = {"title": DOCUMENT_TITLES.get(name, title), "html": rendered}
    return documents
