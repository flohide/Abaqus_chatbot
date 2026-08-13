"""Baut klickbare Deep-Links zu den Original-PDFs für Quellenangaben.

Nutzt file://-URLs mit PDF-Seitenanker (#page=N), den die meisten
PDF-Viewer (Preview.app, Chrome-PDF-Viewer, Adobe Reader) beim Öffnen
respektieren. Der Anker bezieht sich auf die physische Seitenposition in
der Datei, nicht auf das gedruckte Seiten-Label (siehe ingestion/loader.py)
— deshalb wird hier explizit `pdf_page_index` statt `page_number` verwendet.

Hinweis: Manche Browser blockieren/warnen bei file://-Navigation von einer
http(s)://-Seite aus (Sicherheitsrichtlinie, abhängig vom Browser). Im
Terminal (rich-Hyperlinks) funktioniert der Link uneingeschränkt.
"""

from urllib.parse import quote

from .config import settings


def source_pdf_link(source_file: str, pdf_page_index: int) -> str:
    """Baut eine file://-URL zur Original-PDF, die auf der richtigen Seite öffnet."""
    pdf_path = settings.raw_pdf_dir / source_file
    return f"file://{quote(str(pdf_path))}#page={pdf_page_index}"
