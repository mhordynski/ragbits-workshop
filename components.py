import asyncio
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from ragbits.document_search import DocumentSearch
from ragbits.core.vector_stores.qdrant import QdrantVectorStore
from ragbits.core.embeddings.litellm import LiteLLMEmbedder
from ragbits.core.llms.litellm import LiteLLM
from ragbits.core.prompt import Prompt
from ragbits.document_search.documents.document import DocumentType
from ragbits.document_search.ingestion.parsers import DocumentParserRouter
from ragbits.document_search.ingestion.parsers.docling import DoclingDocumentParser
from ragbits.document_search.ingestion.enrichers import ElementEnricherRouter, ElementEnricher
from ragbits.document_search.ingestion.enrichers.image import ImageElementEnricher
from ragbits.document_search.documents.element import ImageElement, TextElement
from ragbits.document_search.retrieval.rephrasers import QueryRephraser


def get_llm():
    return LiteLLM(model_name="gpt-4.1", use_structured_output=True)


def get_document_search():
    vector_store = QdrantVectorStore(
        client=AsyncQdrantClient(url="http://localhost:6333"),
        index_name="workshop",
        embedder=LiteLLMEmbedder(model_name="text-embedding-3-small")
    )
    return DocumentSearch(
        vector_store=vector_store,
        parser_router=DocumentParserRouter({
            DocumentType.PDF: DoclingDocumentParser(ignore_images=True),
        })
    )
