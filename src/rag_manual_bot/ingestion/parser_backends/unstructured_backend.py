"""Parser-Backend: Unstructured (Layout-Modell + Tabellenerkennung, siehe docs/PARSER.md)."""

from pathlib import Path

from unstructured.partition.pdf import partition_pdf


def parse(pdf_path: Path) -> dict[int, str]:
    """Gibt Text pro Seite zurück (1-basiert). Nutzt hi_res-Strategie (Layout-Modell +
    Tabellenerkennung), analog zur höchsten in docs/PARSER.md getesteten Qualitätsstufe."""
    elements = partition_pdf(str(pdf_path), strategy="hi_res", infer_table_structure=True)
    pages: dict[int, list[str]] = {}
    for element in elements:
        if element.text is None:
            # Manche Elementtypen (z. B. reine Bild-/PageBreak-Elemente ohne
            # erkannten Text) liefern None statt "" - auf dem 53-Seiten-Demo-
            # Korpus nie aufgetreten, aber auf dem vollen ~5.100-Seiten-Korpus
            # schon.
            continue
        page_no = element.metadata.page_number or 1
        pages.setdefault(page_no, []).append(element.text)
    return {page_no: "\n".join(texts) for page_no, texts in pages.items()}
