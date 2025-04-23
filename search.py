


import asyncio
from components import get_document_search

async def main():
    document_search = get_document_search()
    result = await document_search.search("What is the friendly interagation room?")

    for element in result:
        print(element.text_representation)

if __name__ == "__main__":
    asyncio.run(main())
