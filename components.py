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



class RemoveWhitespaceElementEnricher(ElementEnricher[TextElement]):
    async def enrich(self, elements: list[TextElement]) -> list[TextElement]:
        """
        Enrich elements.

        Args:
            elements: The elements to be enriched.

        Returns:
            The list of enriched elements.

        Raises:
            EnricherError: If the enrichment of the elements failed.
        """
        new_elements = []
        for element in elements:
            new_elements.append(
                TextElement(
                    content=" ".join(element.text_representation.split()),
                    document_meta=element.document_meta,
                    location=element.location,
                )
            )
        return new_elements
    

class TranslationPromptInput(BaseModel):
    text: str
    input_language: str = "English"
    output_language: str = "Polish"


class TranslationPrompt(Prompt[TranslationPromptInput, str]):
    system_prompt = """
    You are a helpful assistant that translates text from {{ input_language }} to {{ output_language }}.
    Translate the text directly, do not add any additional information.
    """
    user_prompt = """{{text}}"""


class TranslationQueryRephraser(QueryRephraser):
    """
    Rephraser that uses a LLM to rephrase queries.
    """

    async def rephrase(self, query: str) -> list[str]:
        """
        Rephrase a query using the LLM.

        Args:
            query: The query to be rephrased.

        Returns:
            List containing the rephrased query.
        """
        translated = await get_llm().generate(TranslationPrompt(TranslationPromptInput(text=query)))
        return [translated]


def get_document_search():
    vector_store = QdrantVectorStore(
        client=AsyncQdrantClient(url="http://localhost:6333"),
        index_name="workshop",
        embedder=LiteLLMEmbedder(model_name="text-embedding-3-small")
    )
    return DocumentSearch(
        vector_store=vector_store,
        query_rephraser=TranslationQueryRephraser(),
        parser_router=DocumentParserRouter({
            DocumentType.PDF: DoclingDocumentParser(ignore_images=True),
        }),
        enricher_router=ElementEnricherRouter({
            TextElement: RemoveWhitespaceElementEnricher(),
            ImageElement: ImageElementEnricher(llm=get_llm())
        }),
    )
