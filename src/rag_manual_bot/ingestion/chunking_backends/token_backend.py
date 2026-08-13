"""Chunking-Backend: Token-basiertes Splitting (tiktoken) statt Zeichenanzahl.

Wie `recursive_only_backend.py` ohne Markdown-Header-Splitting, aber die
Chunkgrenzen richten sich nach der Tokenanzahl (`cl100k_base`-Encoding, wie
von `gpt-4o-mini`/`text-embedding-3-small` verwendet) statt nach der
Zeichenanzahl - isoliert im Vergleich mit `recursive_only` die
Token-vs-Zeichen-Variable. 200 Tokens/40 Overlap entsprechen etwa den
Zeichen-Defaults (800/150) bei technischem Fließtext (~4 Zeichen/Token).
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..loader import PageDocument

_CHUNK_SIZE_TOKENS = 200
_CHUNK_OVERLAP_TOKENS = 40


def chunk(pages: list[PageDocument]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=_CHUNK_SIZE_TOKENS,
        chunk_overlap=_CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for page in pages:
        for sub_chunk in splitter.split_text(page.text):
            if not sub_chunk.strip():
                continue
            chunks.append(
                Document(
                    page_content=sub_chunk,
                    metadata={
                        "source_file": page.source_file,
                        "page_number": page.page_number,
                        "pdf_page_index": page.pdf_page_index,
                        "section": "",
                    },
                )
            )
    return chunks
