"""RAG-Kette als reine LCEL-Runnables: History-aware Retrieval + Source-Tracking.

LangChain >=1.0 hat `langchain.chains` (inkl. create_retrieval_chain /
create_history_aware_retriever) entfernt. Der Kette wird hier daher explizit
aus LCEL-Bausteinen zusammengesetzt statt einen deprecateten Compat-Layer zu
verwenden.
"""

from operator import itemgetter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from .llms import get_llm
from .prompts import answer_prompt, contextualize_prompt
from .retrieval_backends import RETRIEVAL_BACKENDS


def format_docs(docs: list[Document]) -> str:
    """Baut den Kontext-Block für den Answer-Prompt mit nummerierten Quellenverweisen."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"[Quelle {i}: {meta.get('source_file')}, Seite {meta.get('page_number')}]"
        if meta.get("section"):
            header += f"\n(Abschnitt: {meta['section']})"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_rag_chain(vectorstore: Chroma, llm_provider: str = "openai", retrieval_strategy: str = "mmr") -> Runnable:
    """Baut die vollständige RAG-Kette: Frage -> (History-aware) Retrieval -> Antwort.

    llm_provider steuert, welches Chat-Modell für Contextualize- und
    Answer-Schritt verwendet wird ("openai" oder "mistral") — ermöglicht den
    LLM-Vergleich im Frontend bei unveränderter Wissensbasis.

    retrieval_strategy wählt die Retrieval-Strategie aus
    `rag/retrieval_backends.py::RETRIEVAL_BACKENDS` (Default "mmr" =
    Produktiv-Verhalten, siehe docs/RETRIEVAL.md).
    """
    retrieve = RETRIEVAL_BACKENDS[retrieval_strategy](vectorstore)
    llm = get_llm(llm_provider)

    contextualize_chain = RunnableBranch(
        (lambda x: len(x["chat_history"]) == 0, RunnableLambda(lambda x: x["question"])),
        contextualize_prompt | llm | StrOutputParser(),
    )

    answer_chain = (
        {
            "context": itemgetter("context"),
            "chat_history": itemgetter("chat_history"),
            "question": itemgetter("standalone_question"),
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(standalone_question=contextualize_chain)
        .assign(source_documents=lambda x: retrieve(x["standalone_question"]))
        .assign(context=lambda x: format_docs(x["source_documents"]))
        .assign(answer=answer_chain)
    )
