"""Terminal-Chat-Interface für den Abaqus-Handbuch-Chatbot."""

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

        with console.status("[bold green]Suche im Handbuch ..."):
            try:
                result = chain.invoke({"question": question, "chat_history": chat_history})
            except Exception as exc:  # API-/Netzwerkfehler nicht den Chat abbrechen lassen
                console.print(f"[bold red]Fehler bei der Anfrage:[/bold red] {exc}")
                continue

        console.print("[bold blue]Bot:[/bold blue]")
        console.print(Markdown(result["answer"]))

        sources = _format_sources(result["source_documents"])
        if sources:
            console.print("[dim]Quellen:[/dim]")
            console.print(f"[dim]{sources}[/dim]")
        console.print()

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=result["answer"]))
