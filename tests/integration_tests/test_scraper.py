import pytest
import os
import json
import csv
import io

from langchain_scraperapi.tools import (
    ScraperAPITool,
    ScraperAPIGoogleSearchTool,
    ScraperAPIAmazonSearchTool,
)

API_KEY_AVAILABLE = os.getenv("SCRAPERAPI_API_KEY") is not None
REASON_API_KEY_MISSING = "SCRAPERAPI_API_KEY environment variable not set"

pytestmark = pytest.mark.skipif(not API_KEY_AVAILABLE, reason=REASON_API_KEY_MISSING)


@pytest.fixture(scope="module")
def scraper_tool():
    return ScraperAPITool()

@pytest.fixture(scope="module")
def google_search_tool():
    return ScraperAPIGoogleSearchTool()

@pytest.fixture(scope="module")
def amazon_search_tool():
    return ScraperAPIAmazonSearchTool()


def test_scraper_tool_run_html(scraper_tool):
    """Test scraping a basic HTML page."""
    result = scraper_tool.invoke({"url": "http://example.com"})
    assert isinstance(result, str)
    assert "<!doctype html>" in result.lower()
    assert "<title>example domain</title>" in result.lower()
    assert "this domain is for use in illustrative examples" in result.lower()

def test_scraper_tool_run_text(scraper_tool):
    """Test scraping with text output format."""
    result = scraper_tool.invoke({"url": "http://example.com", "output_format": "text"})
    assert isinstance(result, str)
    assert "example domain" in result.lower()
    assert "this domain is for use in illustrative examples" in result.lower()
    assert "<!doctype html>" not in result.lower() # no html tags


def test_scraper_tool_run_markdown(scraper_tool):
    """Test scraping with markdown output format."""
    result = scraper_tool.invoke({"url": "http://example.com", "output_format": "markdown"})
    assert isinstance(result, str)
    assert "# Example Domain" in result
    assert "illustrative examples" in result
    assert "](https://www.iana.org/domains/example)" in result # check for link markdown


@pytest.mark.asyncio
async def test_scraper_tool_arun_html(scraper_tool):
    """Test async scraping for basic HTML."""
    result = await scraper_tool.ainvoke({"url": "http://example.com"})
    assert isinstance(result, str)
    assert "<!doctype html>" in result.lower()
    assert "<title>example domain</title>" in result.lower()


def test_google_search_tool_run_json(google_search_tool):
    """Test basic Google search with default JSON output."""
    result = google_search_tool.invoke({"query": "what is langchain?"})
    assert isinstance(result, str)
    try:
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "search_information" in data
        assert "organic_results" in data
        assert isinstance(data["organic_results"], list)
        assert len(data["organic_results"]) > 0
    except json.JSONDecodeError:
        pytest.fail("Google search result is not valid JSON")


def test_google_search_tool_run_csv(google_search_tool):
    """Test Google search with CSV output format."""
    result = google_search_tool.invoke({"query": "python programming language", "output_format": "csv"})
    assert isinstance(result, str)
    try:
        csv_file = io.StringIO(result)
        reader = csv.reader(csv_file)
        header = next(reader)
        assert "position" in header or "title" in header
        assert any(row for row in reader)
    except csv.Error:
        pytest.fail("Google search result is not valid CSV")
    except StopIteration:
         pytest.fail("Google search CSV result seems empty (no header or data)")


@pytest.mark.asyncio
async def test_google_search_tool_arun_json(google_search_tool):
    """Test async Google search with JSON output."""
    result = await google_search_tool.ainvoke({"query": "large language models"})
    assert isinstance(result, str)
    try:
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "search_information" in data
        assert "organic_results" in data
    except json.JSONDecodeError:
        pytest.fail("Async Google search result is not valid JSON")


def test_amazon_search_tool_run_json(amazon_search_tool):
    """Test basic Amazon search with default JSON output."""
    result = amazon_search_tool.invoke({"query": "laptop", "tld": "com"})
    assert isinstance(result, str)
    try:
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0
    except json.JSONDecodeError:
        pytest.fail("Amazon search result is not valid JSON")


def test_amazon_search_tool_run_different_tld(amazon_search_tool):
    """Test Amazon search with a different TLD."""
    result = amazon_search_tool.invoke({"query": "tea", "tld": "co.uk"})
    assert isinstance(result, str)
    try:
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0
    except json.JSONDecodeError:
        pytest.fail("Amazon search result is not valid JSON")

@pytest.mark.asyncio
async def test_amazon_search_tool_arun_json(amazon_search_tool):
    """Test async Amazon search with JSON output."""
    result = await amazon_search_tool.ainvoke({"query": "headphones", "tld": "ca"})
    assert isinstance(result, str)
    try:
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0
    except json.JSONDecodeError:
        pytest.fail("Async Amazon search result is not valid JSON")
