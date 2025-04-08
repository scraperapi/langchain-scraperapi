from importlib import metadata

from langchain_scraperapi.chat_models import ChatScraperAPI
from langchain_scraperapi.document_loaders import ScraperAPILoader
from langchain_scraperapi.embeddings import ScraperAPIEmbeddings
from langchain_scraperapi.retrievers import ScraperAPIRetriever
from langchain_scraperapi.toolkits import ScraperAPIToolkit
from langchain_scraperapi.tools import ScraperAPITool
from langchain_scraperapi.vectorstores import ScraperAPIVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatScraperAPI",
    "ScraperAPIVectorStore",
    "ScraperAPIEmbeddings",
    "ScraperAPILoader",
    "ScraperAPIRetriever",
    "ScraperAPIToolkit",
    "ScraperAPITool",
    "__version__",
]
