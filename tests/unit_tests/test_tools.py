from typing import Type

from langchain_scraperapi.tools import (
    ScraperAPITool,
    ScraperAPIGoogleSearchTool,
    ScraperAPIAmazonSearchTool,
)
from langchain_tests.unit_tests import ToolsUnitTests


class TestScraperAPIToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[ScraperAPITool]:
        return ScraperAPITool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "url": "https://example.com",
            "output_format": "text",
            "country_code": "us",
            "device_type": "desktop",
            "premium": False,
            "render": True,
            "keep_headers": False
        }


class TestScraperAPIGoogleSearchToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[ScraperAPIGoogleSearchTool]:
        return ScraperAPIGoogleSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "query": "langchain",
            "tld": "com",
            "country_code": "us",
            "output_format": "json",
            "num": 5,
        }


class TestScraperAPIAmazonSearchToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[ScraperAPIAmazonSearchTool]:
        return ScraperAPIAmazonSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "query": "monitor",
            "tld": "com",
            "country_code": "us",
            "output_format": "json",
            "page": 1,
        }