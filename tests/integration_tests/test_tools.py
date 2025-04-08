from typing import Type

from langchain_scraperapi.tools import (
    ScraperAPITool,
    ScraperAPIGoogleSearchTool,
    ScraperAPIAmazonSearchTool,
)
from langchain_tests.integration_tests import ToolsIntegrationTests


class TestScraperAPIToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[ScraperAPITool]:
        return ScraperAPITool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {
            "url": "https://example.com",
            "output_format": "text",
            "country_code": "us",
            "device_type": "desktop",
            "premium": False,
            "render": True,
            "keep_headers": False
        }


class TestScraperAPIGoogleSearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[ScraperAPIGoogleSearchTool]:
        return ScraperAPIGoogleSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "query": "langchain documentation",
            "output_format": "json",
        }


class TestScraperAPIAmazonSearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[ScraperAPIAmazonSearchTool]:
        return ScraperAPIAmazonSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "query": "standing desk",
            "tld": "com",
        }
    