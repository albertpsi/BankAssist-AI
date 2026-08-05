"""RAG endpoint: basic (FR-L2-9) and enterprise (FR-L3-12) modes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from bankassist.api.schemas import (
    EnterpriseClassificationSummary,
    EnterpriseRagQueryResponse,
    RagQueryRequest,
    RagQueryResponse,
)
from bankassist.config import Settings
from bankassist.llm.factory import build_llm_client
from bankassist.rag.embeddings import OpenAIEmbedder
from bankassist.rag.ingest import chunk_corpus
from bankassist.rag.pipeline import BasicRagPipeline, EnterpriseRagPipeline
from bankassist.rag.stages.bm25_retriever import BM25Retriever
from bankassist.rag.stages.classifier import QueryClassifier
from bankassist.rag.stages.generator import Generator
from bankassist.rag.stages.query_rewriter import QueryRewriter
from bankassist.rag.stages.reranker import CrossEncoderReranker
from bankassist.rag.stages.vector_retriever import VectorRetriever
from bankassist.rag.vector_store import PineconeVectorStore

router = APIRouter(prefix="/rag", tags=["rag"])


def get_pipeline(request: Request) -> BasicRagPipeline:
    """Build the basic pipeline on first use, then reuse it for the app's lifetime.

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
    embedding_cache = getattr(request.app.state, "embedding_cache", None)

    pipeline = BasicRagPipeline(
        settings=settings,
        embedder=OpenAIEmbedder(settings, tracer, embedding_cache=embedding_cache),
        store=PineconeVectorStore(settings, tracer),
        llm=build_llm_client(settings, tracer),
    )
    request.app.state.rag_pipeline = pipeline
    return pipeline


def get_enterprise_pipeline(request: Request) -> EnterpriseRagPipeline:
    """Build the enterprise pipeline on first use, then reuse it (FR-L3-2.2).

    Cached separately from the basic pipeline on ``app.state`` so the two modes
    stay independently selectable — building one never constructs the other.
    """
    cached: EnterpriseRagPipeline | None = getattr(
        request.app.state, "enterprise_rag_pipeline", None
    )
    if cached is not None:
        return cached

    settings: Settings = request.app.state.settings
    tracer = request.app.state.tracer
    llm = build_llm_client(settings, tracer)
    embedding_cache = getattr(request.app.state, "embedding_cache", None)

    _, chunks = chunk_corpus(settings)

    pipeline = EnterpriseRagPipeline(
        settings=settings,
        classifier=QueryClassifier(llm),
        rewriter=QueryRewriter(llm),
        vector_retriever=VectorRetriever(
            OpenAIEmbedder(settings, tracer, embedding_cache=embedding_cache),
            PineconeVectorStore(settings, tracer),
        ),
        bm25_retriever=BM25Retriever(chunks),
        reranker=CrossEncoderReranker(settings.reranker_model),
        generator=Generator(llm),
    )
    request.app.state.enterprise_rag_pipeline = pipeline
    return pipeline


@router.post(
    "/query",
    response_model=RagQueryResponse | EnterpriseRagQueryResponse,
    summary="Answer a banking policy question",
)
def query(
    payload: RagQueryRequest, request: Request
) -> RagQueryResponse | EnterpriseRagQueryResponse:
    """Retrieve relevant policy chunks and answer grounded in them.

    ``mode: "basic"`` (default) is exactly Lab 2's pipeline and response shape
    (NFR-L3-4). ``mode: "enterprise"`` runs the Lab 3 multi-stage pipeline and
    returns the extended, explainable response (FR-L3-12.1).

    Failures propagate as the typed errors ``bankassist.errors`` defines and are
    rendered through the one error envelope every route uses.
    """
    if payload.mode == "enterprise":
        pipeline = get_enterprise_pipeline(request)
        result = pipeline.answer(payload.question)
        return EnterpriseRagQueryResponse(
            answer=result.generated_answer,
            sources=result.citations,
            classification=EnterpriseClassificationSummary(
                label=result.classification.label, confidence=result.classification.confidence
            ),
            rewritten_question=result.rewritten_question,
        )

    pipeline = get_pipeline(request)
    result = pipeline.answer(payload.question)
    return RagQueryResponse(answer=result.answer, sources=result.sources)
