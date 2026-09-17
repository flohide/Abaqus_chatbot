"""Terminal-Chat-Interface für den Abaqus-Handbuch-Chatbot."""

import itertools

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ..citations import source_pdf_link
from ..config import settings
from ..rag.chain import build_rag_chain
from ..rag.vectorstore import load_vectorstore

console = Console()

HELP_TEXT = """\
[bold]Befehle:[/bold]
  /help    diese Hilfe anzeigen
  /reset   Gesprächsverlauf zurücksetzen
  /exit    Chat beenden ([dim]auch: /quit, Strg+D[/dim])
"""


def _format_sources(source_documents) -> str:
    seen = set()
    lines = []
    for doc in source_documents:
        meta = doc.metadata
        key = (meta.get("source_file"), meta.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        label = f"{meta.get('source_file')}, Seite {meta.get('page_number')}"
        if meta.get("pdf_page_index") is not None:
            link = source_pdf_link(meta["source_file"], meta["pdf_page_index"])
            lines.append(f"  • [link={link}]{label}[/link]")
        else:
            lines.append(f"  • {label}")
    return "\n".join(lines)


def _visual_line_count(text: str, width: int) -> int:
    """Anzahl der Terminalzeilen, die `text` beim zeilenweisen Ausgeben mit
    print() belegt (inkl. Zeilenumbruch durch die Terminalbreite) - Grundlage
    dafür, den bereits gedruckten Rohtext-Stream per ANSI-Cursor wieder zu
    löschen und durch die gerenderte Markdown-Fassung zu ersetzen."""
    if width <= 0:
        return text.count("\n") + 1
    total = 0
    for line in text.split("\n"):
        total += max(1, -(-len(line) // width))  # ceil division
    return total


def run_chat() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Abaqus Handbuch-Chatbot[/bold cyan]\n"
            f"Modell: {settings.llm_model} · Wissensbasis: {settings.vectorstore_dir.name}",
            border_style="cyan",
        )
    )
    console.print(HELP_TEXT)

    try:
        vectorstore = load_vectorstore()
    except FileNotFoundError as exc:
        console.print(f"[bold red]Fehler:[/bold red] {exc}")
        return

    chain = build_rag_chain(vectorstore)
    chat_history: list[BaseMessage] = []

    while True:
        try:
            question = console.input("[bold green]Du:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Auf Wiedersehen.[/dim]")
            break

        if not question:
            continue
        if question in ("/exit", "/quit"):
            console.print("[dim]Auf Wiedersehen.[/dim]")
            break
        if question == "/reset":
            chat_history = []
            console.print("[dim]Gesprächsverlauf zurückgesetzt.[/dim]")
            continue
        if question == "/help":
            console.print(HELP_TEXT)
            continue

        stream = chain.stream({"question": question, "chat_history": chat_history})
        try:
            with console.status("[bold green]Suche im Handbuch ..."):
                # Erzeugt noch keine Tokens - liest nur den ersten Chunk an, damit
                # der Spinner bis zum tatsächlichen Start der Antwort läuft.
                first_chunk = next(stream, None)
        except Exception as exc:  # API-/Netzwerkfehler nicht den Chat abbrechen lassen
            console.print(f"[bold red]Fehler bei der Anfrage:[/bold red] {exc}")
            continue

        console.print("[bold blue]Bot:[/bold blue]")
        source_documents: list = []
        answer_parts: list[str] = []
        remaining = itertools.chain([first_chunk] if first_chunk is not None else [], stream)
        stream_error: Exception | None = None
        try:
            for chunk in remaining:
                if "source_documents" in chunk:
                    source_documents = chunk["source_documents"]
                if "answer" in chunk:
                    token = chunk["answer"]
                    answer_parts.append(token)
                    print(token, end="", flush=True)  # rohe Tokens für sofortige Anzeige waehrend des Streamens
        except Exception as exc:
            # Bereits gestreamte Tokens bleiben erhalten (siehe answer_parts
            # unten) statt verworfen zu werden - sonst laufen sichtbarer
            # Chatverlauf und chat_history auseinander.
            stream_error = exc
        print()

        answer = "".join(answer_parts)

        if answer and console.is_terminal:
            # Rohtext-Stream durch eine einmalige Markdown-Renderung ersetzen,
            # statt ihn dauerhaft als Rohtext stehen zu lassen (Trade-off aus
            # Punkt 5: waehrend des Streamens weiterhin Rohtext, da Markdown-
            # Rendering den vollstaendigen Text voraussetzt).
            printed_lines = _visual_line_count(answer, console.width)
            console.file.write(f"\033[{printed_lines}A\033[J")
            console.file.flush()
            console.print(Markdown(answer))

        if stream_error is not None:
            console.print(f"[bold red]Fehler bei der Anfrage:[/bold red] {stream_error}")

        sources = _format_sources(source_documents)
        if sources:
            console.print("[dim]Quellen:[/dim]")
            console.print(f"[dim]{sources}[/dim]")
        console.print()

        chat_history.append(HumanMessage(content=question))
        if answer:
            chat_history.append(AIMessage(content=answer))
