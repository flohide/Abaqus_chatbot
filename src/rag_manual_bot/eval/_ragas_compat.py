"""Kompatibilitäts-Shim für ragas 0.4.3 unter LangChain >=1.0.

ragas importiert unbedingt `langchain_community.chat_models.vertexai`
(`ChatVertexAI`) beim Laden von `ragas.llms`. Dieses Submodul wurde im Zuge
des "Sunsetting" von langchain-community (siehe Deprecation-Warnung beim
Import) aus Version 0.4.2 entfernt — unabhängig davon, ob Google
Vertex AI überhaupt genutzt wird, bricht der reine Import von `ragas`
dadurch mit `ModuleNotFoundError`. Wir nutzen in diesem Projekt weder
Vertex AI noch diesen Codepfad von ragas; ein minimaler Platzhalter genügt,
um den Import zu befriedigen.

Muss vor JEDEM `import ragas` (bzw. `ragas.llms`) ausgeführt werden, z. B.
ganz am Anfang jedes Eval-Skripts:

    from rag_manual_bot.eval import _ragas_compat  # noqa: F401  (vor ragas-Import!)
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # Platzhalter, wird in diesem Projekt nie instanziiert
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ChatVertexAI ist in diesem Projekt nur ein Kompatibilitäts-Stub "
                "(siehe rag_manual_bot/eval/_ragas_compat.py) und nicht nutzbar."
            )

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
