import asyncio
from components import get_document_search


async def main():
    document_search = get_document_search()
    await document_search.ingest(
        "https://https://edukacja.fdds.pl/pluginfile.php/92756/mod_resource/content/2/Bede-swiadkiem-Poradnik-dla-nastolatkow-uczestniczacych-w-procedurach-karnych_2016.pdf"
    )


if __name__ == "__main__":
    asyncio.run(main())