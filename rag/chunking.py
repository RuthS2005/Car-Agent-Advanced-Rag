"""Splits documents into chunks. Each chunk keeps its own text and a copy of the document's metadata."""
from dataclasses import dataclass

from rag.loader import Doc


@dataclass
class Chunk:
    id: str            # "<doc id>#<number>", e.g. "of-210-city#2"
    doc_id: str
    text: str          # the original text, exactly as it will be shown to the model
    meta: dict


def chunk_fixed(doc: Doc, size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Baseline: cut every `size` characters, with `overlap` characters repeated between neighbours."""
    chunks = []
    step = size - overlap
    for start in range(0, max(len(doc.text), 1), step):
        piece = doc.text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, piece, dict(doc.meta)))
        if start + size >= len(doc.text):
            break
    return chunks


def chunk_by_headings(doc: Doc) -> list[Chunk]:
    """Split a document into one chunk per "## " section.

    - Every chunk starts with a heading path line: "<document title> › <section heading>".
      The text before the first "## " goes in a chunk whose path line is just the title.
    - Skip the "# " title line itself (the path line already carries the title).
    - Keep each section whole, so a table stays together with its header row.
    - Copy the document metadata and add meta["section"] = the section heading.
    - Ids: f"{doc.id}#{n}", numbered from 0.
    """
    chunks = []
    section = ""
    content = []

    def add_chunk() -> None:
        body = "\n".join(line for line in content if not line.startswith("# ")).strip()
        if not body and not section:
            return

        path = f"{doc.title} › {section}" if section else doc.title
        text = f"{path}\n{body}" if body else path
        meta = dict(doc.meta)
        meta["section"] = section
        chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, text, meta))

    for line in doc.text.splitlines():
        if line.startswith("## "):
            add_chunk()
            section = line[3:].strip()
            content = []
        else:
            content.append(line)
    add_chunk()
    return chunks
