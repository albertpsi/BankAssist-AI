"""Basic RAG endpoint (FR-L2-9)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from bankassist.api.schemas import RagQueryRequest, RagQueryResponse
from bankassist.config import Settings
from bankassist.llm.factory import build_llm_client
from bankassist.rag.embeddings import OpenAIEmbedder
from bankassist.rag.pipeline import BasicRagPipeline
from bankassist.rag.vector_store import PineconeVectorStore

router = APIRouter(prefix="/rag", tags=["rag"])


def get_pipeline(request: Request) -> BasicRagPipeline:
    """Build the pipeline on first use, then reuse it for the app's lifetime.

    FR-L2-9.4: constructing ``PineconeVectorStore`` is what actually requires
    ``PINECONE_API_KEY``. Doing that here rather than at app startup means
    ``/health`` and every other route stay usable on a machine with no Pinecone
    account configured — only a call to this route needs the credential.
    """
    cached: BasicRagPipeline | None = getattr(request.app.state, "rag_pipeline", None)
    if cached is not None:
        return cached

    settings: Settings = request.app.state.settings
    tracer = request.app.state.tracer

    pipeline = BasicRagPipeline(
        settings=settings,
        embedder=OpenAIEmbedder(settings, tracer),
        store=PineconeVectorStore(settings, tracer),
        llm=build_llm_client(settings, tracer),
    )
    request.app.state.rag_pipeline = pipeline
    return pipeline


@router.post("/query", response_model=RagQueryResponse, summary="Answer a banking policy question")
def query(payload: RagQueryRequest, request: Request) -> RagQueryResponse:
    """Retrieve relevant policy chunks and answer grounded in them (FR-L2-9.1).

    Failures propagate as the typed errors ``bankassist.errors`` defines —
    ``ConfigurationError``, ``EmbeddingError``, ``VectorStoreError``, ``LLMError``
    — and are rendered through the one error envelope every route uses.
    """
    pipeline = get_pipeline(request)
    result = pipeline.answer(payload.question)
    return RagQueryResponse(answer=result.answer, sources=result.sources)
