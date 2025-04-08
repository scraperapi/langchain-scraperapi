"""Test ScraperAPI embeddings."""

from typing import Type

from langchain_scraperapi.embeddings import ScraperAPIEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[ScraperAPIEmbeddings]:
        return ScraperAPIEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
