from ragbits.core.prompt import Prompt
from ragbits.core.llms.litellm import LiteLLM
from pydantic import BaseModel
import asyncio

class QueryWithContext(BaseModel):
    """
    Input format for the QueryWithContext.
    """

    query: str


class BuyRequest(BaseModel):
    """
    Output format for the BuyRequest.
    """

    product_name: str
    quantity: int

class RAGPrompt(Prompt[QueryWithContext, BuyRequest]):
    """
    A simple prompt for RAG system.
    """

    system_prompt = """
    Your task is extract information from user query about their buy request.
    """

    user_prompt = """
    User query:
    {{ query }}
    """


async def main():
    llm = LiteLLM(model_name="gpt-4.1", use_structured_output=True)
    query = "I want to buy two kebabs"
    prompt = RAGPrompt(QueryWithContext(query=query))
    response = await llm.generate(prompt)

    for i in range(response.quantity):
        print("This is " + response.product_name + " for you!")


asyncio.run(main())


