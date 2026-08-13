"""Parser-Backend: Docling (Deep-Learning-Layoutanalyse, siehe docs/PARSER.md)."""

from pathlib import Path

from docling.document_converter import DocumentConverter


def parse(pdf_path: Path) -> dict[int, str]:
    """Gibt Markdown-Text pro Seite zurück (1-basiert)."""
    result_doc = DocumentConverter().convert(str(pdf_path)).document
    return {
        page_no: result_doc.export_to_markdown(page_no=page_no)
        for page_no in range(1, result_doc.num_pages() + 1)
    }
