import pytest
from unittest.mock import MagicMock, patch, AsyncMock

import aiohttp
import requests
from pydantic import ValidationError

from langchain_scraperapi.utils import (
    ScraperAPIWrapper,
    ScraperAPIStructuredWrapper,
    SCRAPERAPI_BASE_URL,
    SCRAPERAPI_STRUCTURED_BASE_URL,
)
from langchain_scraperapi.tools import (
    ScraperAPITool,
    ScraperAPIGoogleSearchTool,
    ScraperAPIAmazonSearchTool,
    ScraperAPIToolInput,
    ScraperAPIGoogleSearchToolInput,
    ScraperAPIAmazonSearchToolInput,
)


pytestmark = pytest.mark.allow_hosts("127.0.0.1")

@pytest.fixture
def mock_env_api_key(monkeypatch):
    monkeypatch.setenv("SCRAPERAPI_API_KEY", "test_api_key")

@pytest.fixture
def scraper_api_wrapper(mock_env_api_key):
    return ScraperAPIWrapper()

@pytest.fixture
def scraper_api_structured_wrapper(mock_env_api_key):
    return ScraperAPIStructuredWrapper()

# --- Test ScraperAPIWrapper ---

def test_scraper_api_wrapper_init_from_env(mock_env_api_key):
    wrapper = ScraperAPIWrapper()
    assert wrapper.scraperapi_api_key.get_secret_value() == "test_api_key"

def test_scraper_api_wrapper_init_direct():
    wrapper = ScraperAPIWrapper(scraperapi_api_key="direct_key")
    assert wrapper.scraperapi_api_key.get_secret_value() == "direct_key"

def test_scraper_api_wrapper_init_missing_key(monkeypatch):
    monkeypatch.delenv("SCRAPERAPI_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="Did not find scraperapi_api_key"):
        ScraperAPIWrapper()

@patch('requests.get')
def test_scraper_api_wrapper_scrape_success(mock_get, scraper_api_wrapper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>Success</html>"
    mock_get.return_value = mock_response

    result = scraper_api_wrapper.scrape(url="http://example.com")

    mock_get.assert_called_once_with(
        SCRAPERAPI_BASE_URL,
        params={'api_key': 'test_api_key', 'url': 'http://example.com'}
    )
    mock_response.raise_for_status.assert_called_once()
    assert result == "<html>Success</html>"

@patch('requests.get')
def test_scraper_api_wrapper_scrape_with_params(mock_get, scraper_api_wrapper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "Success Text"
    mock_get.return_value = mock_response

    result = scraper_api_wrapper.scrape(
        url="http://example.com",
        output_format="text",
        country_code="us",
        device_type="mobile",
        premium=True,
        render=False,
        keep_headers=True,
    )

    expected_params = {
        'api_key': 'test_api_key',
        'url': 'http://example.com',
        'output_format': 'text',
        'country_code': 'us',
        'device_type': 'mobile',
        'premium': 'true',
        'render': 'false',
        'keep_headers': 'true',
    }
    mock_get.assert_called_once_with(SCRAPERAPI_BASE_URL, params=expected_params)
    mock_response.raise_for_status.assert_called_once()
    assert result == "Success Text"

@patch('requests.get')
def test_scraper_api_wrapper_scrape_http_error(mock_get, scraper_api_wrapper):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Not Found")
    mock_get.return_value = mock_response

    with pytest.raises(requests.exceptions.HTTPError, match="Not Found"):
        scraper_api_wrapper.scrape(url="http://example.com/404")

    mock_get.assert_called_once_with(
        SCRAPERAPI_BASE_URL,
        params={'api_key': 'test_api_key', 'url': 'http://example.com/404'}
    )
    mock_response.raise_for_status.assert_called_once()

@patch('requests.get')
def test_scraper_api_wrapper_scrape_connection_error(mock_get, scraper_api_wrapper):
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

    with pytest.raises(requests.exceptions.ConnectionError, match="Connection failed"):
        scraper_api_wrapper.scrape(url="http://unreachable.com")

    mock_get.assert_called_once_with(
        SCRAPERAPI_BASE_URL,
        params={'api_key': 'test_api_key', 'url': 'http://unreachable.com'}
    )

@pytest.mark.asyncio
async def test_scraper_api_wrapper_scrape_async_success(scraper_api_wrapper):
    
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="<html>Async Success</html>")
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_response_context_manager = AsyncMock()
    mock_response_context_manager.__aenter__.return_value = mock_response
    mock_response_context_manager.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get.return_value = mock_response_context_manager
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch('langchain_scraperapi.utils.aiohttp.ClientSession', return_value=mock_session) as mock_clientsession_constructor:
        result = await scraper_api_wrapper.scrape_async(url="http://example.com")

    mock_clientsession_constructor.assert_called_once_with()
    mock_session.__aenter__.assert_awaited_once()
    mock_session.__aexit__.assert_awaited_once()

    expected_params = {
        'api_key': 'test_api_key',
        'url': 'http://example.com'
    }
    mock_session.get.assert_called_once_with(SCRAPERAPI_BASE_URL, params=expected_params)

    mock_response_context_manager.__aenter__.assert_awaited_once()
    mock_response_context_manager.__aexit__.assert_awaited_once()

    mock_response.raise_for_status.assert_not_called()
    mock_response.text.assert_awaited_once()

    assert result == "<html>Async Success</html>"

@pytest.mark.asyncio

async def test_scraper_api_wrapper_scrape_async_with_params(scraper_api_wrapper):

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="Async Success Text")
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_response_context_manager = AsyncMock()
    mock_response_context_manager.__aenter__.return_value = mock_response
    mock_response_context_manager.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get.return_value = mock_response_context_manager
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch('langchain_scraperapi.utils.aiohttp.ClientSession', return_value=mock_session) as mock_clientsession_constructor:
        result = await scraper_api_wrapper.scrape_async(
            url="http://example.com",
            output_format="text",
            country_code="ca",
            device_type="desktop",
            premium=False,
        render=True,
        keep_headers=False,
    )
        
    mock_clientsession_constructor.assert_called_once_with()
    mock_session.__aenter__.assert_awaited_once()
    mock_session.__aexit__.assert_awaited_once()

    expected_params = {
        'api_key': 'test_api_key',
        'url': 'http://example.com',
        'output_format': 'text',
        'country_code': 'ca',
        'device_type': 'desktop',
        'premium': 'false',
        'render': 'true',
        'keep_headers': 'false',
    }
    mock_session.get.assert_called_once_with(SCRAPERAPI_BASE_URL, params=expected_params)

    mock_response_context_manager.__aenter__.assert_awaited_once()
    mock_response_context_manager.__aexit__.assert_awaited_once()

    mock_response.raise_for_status.assert_not_called()
    mock_response.text.assert_awaited_once()

    assert result == "Async Success Text"

@pytest.mark.asyncio

async def test_scraper_api_wrapper_scrape_async_http_error(scraper_api_wrapper):

    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 500
    mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
        MagicMock(), (), status=500, message="Server Error"
    )
    mock_response.text = AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_response_context_manager = AsyncMock()
    mock_response_context_manager.__aenter__.return_value = mock_response
    mock_response_context_manager.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get.return_value = mock_response_context_manager
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch('langchain_scraperapi.utils.aiohttp.ClientSession', return_value=mock_session) as mock_clientsession_constructor:
        with pytest.raises(aiohttp.ClientResponseError, match="Server Error"):
            await scraper_api_wrapper.scrape_async(url="http://example.com/500")

    mock_clientsession_constructor.assert_called_once_with()
    mock_session.__aenter__.assert_awaited_once()
    mock_session.__aexit__.assert_awaited_once()

    expected_params = {
        'api_key': scraper_api_wrapper.scraperapi_api_key.get_secret_value(),
        'url': 'http://example.com/500'
    }
    mock_session.get.assert_called_once_with(SCRAPERAPI_BASE_URL, params=expected_params)

    mock_response_context_manager.__aenter__.assert_awaited_once()
    mock_response.raise_for_status.assert_called_once()
    mock_response.text.assert_not_awaited()
    mock_response_context_manager.__aexit__.assert_awaited_once()


# --- Test ScraperAPIStructuredWrapper ---

def test_scraper_api_structured_wrapper_init(mock_env_api_key):
    wrapper = ScraperAPIStructuredWrapper()
    assert wrapper.scraperapi_api_key.get_secret_value() == "test_api_key"


@patch('requests.get')
def test_structured_wrapper_google_search(mock_get, scraper_api_structured_wrapper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"google": "results"}'
    mock_get.return_value = mock_response

    result = scraper_api_structured_wrapper.google_search(
        query="pizza recipe",
        country_code="us",
        tld="com",
        output_format="json",
        num=10
    )

    expected_params = {
        'api_key': 'test_api_key',
        'query': 'pizza recipe',
        'country_code': 'us',
        'tld': 'com',
        'output_format': 'json',
        'num': 10,
    }
    mock_get.assert_called_once_with(
         f"{SCRAPERAPI_STRUCTURED_BASE_URL}google/search", params=expected_params
    )
    assert result == '{"google": "results"}'


@patch('requests.get')
def test_structured_wrapper_amazon_search(mock_get, scraper_api_structured_wrapper):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"amazon": "results"}'
    mock_get.return_value = mock_response

    result = scraper_api_structured_wrapper.amazon_search(
        query="shoes",
        country_code="uk",
        tld="co.uk",
        page=2
    )

    expected_params = {
        'api_key': 'test_api_key',
        'query': 'shoes',
        'country': 'uk',
        'tld': 'co.uk',
        'page': 2,
    }
    mock_get.assert_called_once_with(
         f"{SCRAPERAPI_STRUCTURED_BASE_URL}amazon/search", params=expected_params
    )
    assert result == '{"amazon": "results"}'


@pytest.mark.asyncio

async def test_structured_wrapper_google_search_async(scraper_api_structured_wrapper):

    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value='{"async_google": "results"}')
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_response_context_manager = AsyncMock()
    mock_response_context_manager.__aenter__.return_value = mock_response
    mock_response_context_manager.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get.return_value = mock_response_context_manager
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch('langchain_scraperapi.utils.aiohttp.ClientSession', return_value=mock_session) as mock_clientsession_constructor:
        result = await scraper_api_structured_wrapper.google_search_async(
            query="async pizza",
            tld="fr",
            num=5
        )

    mock_clientsession_constructor.assert_called_once_with()
    mock_session.__aenter__.assert_awaited_once()
    mock_session.__aexit__.assert_awaited_once()

    expected_params = {
        'api_key': scraper_api_structured_wrapper.scraperapi_api_key.get_secret_value(),
        'query': 'async pizza',
        'tld': 'fr',
        'num': 5,
    }
    expected_url = f"{SCRAPERAPI_STRUCTURED_BASE_URL}google/search"
    mock_session.get.assert_called_once_with(expected_url, params=expected_params)

    mock_response_context_manager.__aenter__.assert_awaited_once()
    mock_response_context_manager.__aexit__.assert_awaited_once()

    mock_response.raise_for_status.assert_not_called()
    mock_response.text.assert_awaited_once()

    assert result == '{"async_google": "results"}'


@pytest.mark.asyncio

async def test_structured_wrapper_amazon_search_async(scraper_api_structured_wrapper):

    mock_response = AsyncMock(spec=aiohttp.ClientResponse)
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value='{"async_amazon": "results"}')
    mock_response.raise_for_status = MagicMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_response_context_manager = AsyncMock()
    mock_response_context_manager.__aenter__.return_value = mock_response
    mock_response_context_manager.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.get.return_value = mock_response_context_manager
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__ = AsyncMock(return_value=None)

    with patch('langchain_scraperapi.utils.aiohttp.ClientSession.get', return_value=mock_response_context_manager) as mock_get:
        result = await scraper_api_structured_wrapper.amazon_search_async(
            query="async shoes",
            country_code="ca",
            tld="ca",
            page=1
        )

    expected_params = {
        'api_key': 'test_api_key',
        'query': 'async shoes',
        'country': 'ca',
        'tld': 'ca',
        'page': 1,
    }
    expected_url = f"{SCRAPERAPI_STRUCTURED_BASE_URL}amazon/search"
    mock_get.assert_called_once_with(expected_url, params=expected_params)

    mock_response_context_manager.__aenter__.assert_awaited_once()
    mock_response_context_manager.__aexit__.assert_awaited_once()

    mock_response.raise_for_status.assert_not_called()
    mock_response.text.assert_awaited_once()

    assert result == '{"async_amazon": "results"}'

# --- Test ScraperAPITool ---

@pytest.fixture
def scraper_tool(mock_env_api_key):
    return ScraperAPITool()

def test_scraper_tool_attributes(scraper_tool):
    assert scraper_tool.name == "scraperapi"
    assert "scraping web content" in scraper_tool.description
    assert scraper_tool.args_schema == ScraperAPIToolInput

@patch.object(ScraperAPIWrapper, 'scrape')
def test_scraper_tool_run(mock_scrape, scraper_tool):
    mock_scrape.return_value = "Scraped content"
    result = scraper_tool._run(
        url="http://test.com",
        output_format="text",
        country_code="gb"
    )
    mock_scrape.assert_called_once_with(
        url="http://test.com",
        output_format="text",
        country_code="gb",
        device_type=None,
        premium=None,
        render=None,
        keep_headers=None,
    )
    assert result == "Scraped content"

@patch.object(ScraperAPIWrapper, 'scrape')
def test_scraper_tool_run_error(mock_scrape, scraper_tool):
    mock_scrape.side_effect = ValueError("API failed")
    result = scraper_tool._run(url="http://fail.com")
    assert result == "Error: API failed"
    mock_scrape.assert_called_once()


@pytest.mark.asyncio

@patch.object(ScraperAPIWrapper, 'scrape_async', new_callable=AsyncMock)
async def test_scraper_tool_arun(mock_scrape_async, scraper_tool):
    mock_scrape_async.return_value = "Async scraped content"
    result = await scraper_tool._arun(
        url="http://async-test.com",
        premium=True,
        render=True
    )
    mock_scrape_async.assert_called_once_with(
        url="http://async-test.com",
        output_format=None,
        country_code=None,
        device_type=None,
        premium=True,
        render=True,
        keep_headers=None,
    )
    assert result == "Async scraped content"

@pytest.mark.asyncio

@patch.object(ScraperAPIWrapper, 'scrape_async', new_callable=AsyncMock)
async def test_scraper_tool_arun_error(mock_scrape_async, scraper_tool):
    mock_scrape_async.side_effect = ConnectionError("Async connection failed")
    result = await scraper_tool._arun(url="http://async-fail.com")
    assert result == "Error: Async connection failed"
    mock_scrape_async.assert_called_once_with(
        url="http://async-fail.com",
        output_format=None,
        country_code=None,
        device_type=None,
        premium=None,
        render=None,
        keep_headers=None,
    )


# --- Test ScraperAPIGoogleSearchTool ---

@pytest.fixture
def google_search_tool(mock_env_api_key):
    return ScraperAPIGoogleSearchTool()

def test_google_search_tool_attributes(google_search_tool):
    assert google_search_tool.name == "scraperapi_google_search"
    assert "Google searches" in google_search_tool.description
    assert google_search_tool.args_schema == ScraperAPIGoogleSearchToolInput

@patch.object(ScraperAPIStructuredWrapper, 'google_search')
def test_google_search_tool_run(mock_search, google_search_tool):
    mock_search.return_value = '{"google_data": "found"}'
    result = google_search_tool._run(
        query="test search",
        tld="de",
        output_format="json"
    )
    mock_search.assert_called_once_with(
        query="test search",
        country_code=None,
        tld="de",
        output_format="json",
        uule=None,
        num=None,
        hl=None,
        gl=None,
        ie=None,
        oe=None,
        start=None,
    )
    assert result == '{"google_data": "found"}'

@patch.object(ScraperAPIStructuredWrapper, 'google_search')
def test_google_search_tool_run_error(mock_search, google_search_tool):
    mock_search.side_effect = RuntimeError("Search failed")
    result = google_search_tool._run(query="failing search")
    assert result == "Error: Search failed"
    mock_search.assert_called_once()

@pytest.mark.asyncio

@patch.object(ScraperAPIStructuredWrapper, 'google_search_async', new_callable=AsyncMock)
async def test_google_search_tool_arun(mock_search_async, google_search_tool):
    mock_search_async.return_value = '{"async_google_data": "found"}'
    result = await google_search_tool._arun(
        query="async test search",
        num=5,
        hl="fr"
    )
    mock_search_async.assert_called_once_with(
        query="async test search",
        country_code=None,
        tld=None,
        output_format=None,
        uule=None,
        num=5,
        hl="fr",
        gl=None,
        ie=None,
        oe=None,
        start=None,
    )
    assert result == '{"async_google_data": "found"}'

@pytest.mark.asyncio

@patch.object(ScraperAPIStructuredWrapper, 'google_search_async', new_callable=AsyncMock)
async def test_google_search_tool_arun_error(mock_search_async, google_search_tool):
    mock_search_async.side_effect = Exception("Async Search failed")
    result = await google_search_tool._arun(query="async failing search")
    assert result == "Error: Async Search failed"
    mock_search_async.assert_called_once_with(
        query="async failing search",
        country_code=None,
        tld=None,
        output_format=None,
        uule=None,
        num=None,
        hl=None,
        gl=None,
        ie=None,
        oe=None,
        start=None,
    )

# --- Test ScraperAPIAmazonSearchTool ---

@pytest.fixture
def amazon_search_tool(mock_env_api_key):
    return ScraperAPIAmazonSearchTool()

def test_amazon_search_tool_attributes(amazon_search_tool):
    assert amazon_search_tool.name == "scraperapi_amazon_search"
    assert "Amazon searches" in amazon_search_tool.description
    assert amazon_search_tool.args_schema == ScraperAPIAmazonSearchToolInput

@patch.object(ScraperAPIStructuredWrapper, 'amazon_search')
def test_amazon_search_tool_run(mock_search, amazon_search_tool):
    mock_search.return_value = '{"amazon_data": "found"}'
    result = amazon_search_tool._run(
        query="amazon product",
        country_code="es",
        tld="es",
        page=3
    )
    mock_search.assert_called_once_with(
        query="amazon product",
        country_code="es",
        tld="es",
        output_format=None,
        page=3,
    )
    assert result == '{"amazon_data": "found"}'

@patch.object(ScraperAPIStructuredWrapper, 'amazon_search')
def test_amazon_search_tool_run_error(mock_search, amazon_search_tool):
    mock_search.side_effect = TypeError("Bad Amazon")
    result = amazon_search_tool._run(query="failing amazon")
    assert result == "Error: Bad Amazon"
    mock_search.assert_called_once()

@pytest.mark.asyncio

@patch.object(ScraperAPIStructuredWrapper, 'amazon_search_async', new_callable=AsyncMock)
async def test_amazon_search_tool_arun(mock_search_async, amazon_search_tool):
    mock_search_async.return_value = '{"async_amazon_data": "found"}'
    result = await amazon_search_tool._arun(
        query="async amazon product",
        tld="com.au",
        output_format="csv"
    )
    mock_search_async.assert_called_once_with(
        query="async amazon product",
        country_code=None,
        tld="com.au",
        output_format="csv",
        page=None,
    )
    assert result == '{"async_amazon_data": "found"}'

@pytest.mark.asyncio

@patch.object(ScraperAPIStructuredWrapper, 'amazon_search_async', new_callable=AsyncMock)
async def test_amazon_search_tool_arun_error(mock_search_async, amazon_search_tool):
    mock_search_async.side_effect = TimeoutError("Amazon Timeout")
    result = await amazon_search_tool._arun(query="timeout amazon")
    assert result == "Error: Amazon Timeout"
    mock_search_async.assert_called_once_with(
        query="timeout amazon",
        country_code=None,
        tld=None,
        output_format=None,
        page=None,
    )
