"""Test chat model integration."""

from typing import Type

from langchain_scraperapi.chat_models import ChatScraperAPI
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatScraperAPIUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatScraperAPI]:
        return ChatScraperAPI

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
