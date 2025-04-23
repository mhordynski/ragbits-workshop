from ragbits.chat.interface import ChatInterface
from ragbits.chat.interface.types import ChatResponse
from ragbits.core.prompt import ChatFormat
from typing import AsyncGenerator

from components import get_document_search, get_llm

from ragbits.core.prompt import Prompt
from ragbits.core.llms.litellm import LiteLLM
from pydantic import BaseModel

class QueryWithContext(BaseModel):
    """
    Input format for the QueryWithContext.
    """

    query: str
    context: list[str]


class RAGPrompt(Prompt[QueryWithContext]):
    """
    A simple prompt for RAG system.
    """

    system_prompt = """
    You are a helpful assistant. Answer the QUESTION that will be provided using CONTEXT.
    If in the given CONTEXT there is not enough information refuse to answer.
    """

    user_prompt = """
    QUESTION:
    {{ query }}

    CONTEXT:
    {% for item in context %}
        {{ item }}
    {% endfor %}
    """


class MyChat(ChatInterface):
    
    async def chat(
        self,
        message: str,
        history: ChatFormat | None = None,
        context: dict | None = None,
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        Process a chat message and yield responses asynchronously.

        Args:
            message: The current user message
            history: Optional list of previous messages in the conversation
            context: Optional context

        Yields:
            ChatResponse objects containing different types of content:
            - Text chunks for the actual response
            - Reference documents used to generate the response

        Example:
            ```python
            chat = MyChatImplementation()
            async for response in chat.chat("What is Python?"):
                if text := response.as_text():
                    print(f"Text: {text}")
                elif ref := response.as_reference():
                    print(f"Reference: {ref.title}")
                    pass
        """

        document_search = get_document_search()
        result = await document_search.search(message)

        async for response in get_llm().generate_streaming(RAGPrompt(
            QueryWithContext(query=message, context=[x.text_representation for x in result]))):
            yield self.create_text_response(response)