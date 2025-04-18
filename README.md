# langchain-scraperapi

This package contains the LangChain integration with ScraperAPI

## Installation

```bash
pip install -U langchain-scraperapi
```

And you should configure credentials by setting the environment variable `SCRAPERAPI_API_KEY`.

## Tools

### ScraperAPITool

`ScraperAPITool` exposes the web scraping tool from ScraperAPI.

```python
from langchain_scraperapi import ScraperAPITool

tool = ScraperAPITool()
tool.invoke("url: http://example.com", "output_format": "markdown")
```

### ScraperAPIGoogleSearchTool

`ScraperAPIGoogleSearchTool` allows the scraping of Google search results in `json` or `csv` format.

```python
from langchain_scraperapi import ScraperAPIGoogleSearchTool

tool = ScraperAPITool()
tool.invoke("query": "What is ScraperAPI?")
```

### ScraperAPIAmazonSearchTool

`ScraperAPIAmazonSearchTool` allows the scraping of Amazon search results in `json` or `csv` format.

```python
from langchain_scraperapi import ScraperAPIAmazonSearchTool

tool = ScraperAPITool()
tool.invoke("query": "office chairs", "output_format": "csv")
```