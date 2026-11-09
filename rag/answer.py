"""Answers a question from the retrieved chunks, as a Pydantic object with citations."""
from pydantic import BaseModel, Field

from rag.chunking import Chunk

MODEL = "claude-haiku-5-5"

SYSTEM = """אתה עונה על שאלות של לקוחות חברת הרכב "אופק רכב".
ענה רק על סמך הקטעים שבתוך <chunk>. אל תשתמש בידע כללי.
לכל טענה בתשובה, הוסף ציטוט: מזהה הקטע וציטוט מדויק מתוכו, מילה במילה.
אם הקטעים לא מספיקים כדי לענות, החזר insufficient_context=true, בלי ציטוטים.
ענה בעברית, במשפטים קצרים."""


class Citation(BaseModel):
    chunk_id: str = Field(description="The chunk id, exactly as it appears in <chunk id=...>")
    quote: str = Field(description="An exact quote from that chunk, word for word")


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    insufficient_context: bool


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f'<chunk id="{c.id}">\n{c.text}\n</chunk>' for c in chunks)


def ask(question: str, chunks: list[Chunk], client=None) -> tuple[Answer, object]:
    """Returns the parsed Answer and the token usage. Needs ANTHROPIC_API_KEY in the environment."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"{build_context(chunks)}\n\nשאלה: {question}"}],
        output_format=Answer,
    )
    return response.parsed_output, response.usage


def verify_citations(answer: Answer, chunks: list[Chunk]) -> list[Citation]:
    """Return citations that are not backed by the supplied chunks.

    A citation is bad if its chunk_id is not one of the chunks we sent,
    or if its quote does not appear in that chunk's text. Whitespace
    differences, including line breaks, are ignored.
    An empty list means every citation checks out.
    """
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    invalid = []
    for citation in answer.citations:
        chunk = chunks_by_id.get(citation.chunk_id)
        if chunk is None or " ".join(citation.quote.split()) not in " ".join(chunk.text.split()):
            invalid.append(citation)
    return invalid
