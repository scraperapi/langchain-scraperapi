# LangChain – ScraperAPI

Give your AI agent the ability to browse websites, search Google and Amazon in just two lines of code.

The `langchain-scraperapi` package adds three ready-to-use LangChain tools backed by the [ScraperAPI](https://www.scraperapi.com/) service:

| Tool class | Use it to |
|------------|----------|
| `ScraperAPITool` | Grab the HTML/text/markdown of any web page |
| `ScraperAPIGoogleSearchTool` | Get structured Google Search SERP data |
| `ScraperAPIAmazonSearchTool` | Get structured Amazon product-search data |

## Installation

```bash
pip install -U langchain-scraperapi
```

## Setup

Create an account at https://www.scraperapi.com/ and get an API key, then set it as an environment variable:

```python
import os
os.environ["SCRAPERAPI_API_KEY"] = "your-api-key"
```

## Quick Start

### ScraperAPITool — Browse any website

Scrape HTML, text, or markdown from any webpage:

```python
from langchain_scraperapi.tools import ScraperAPITool

tool = ScraperAPITool()

# Get text content
result = tool.invoke({
    "url": "https://example.com",
    "output_format": "text",
    "render": True
})
print(result)
```

**Parameters:**
- `url` (required) – target page URL
- `output_format` – `"text"` | `"markdown"` (default returns HTML)
- `country_code` – e.g. `"us"`, `"de"`
- `device_type` – `"desktop"` | `"mobile"`
- `premium` – use premium proxies
- `render` – run JavaScript before returning content
- `keep_headers` – include response headers

### ScraperAPIGoogleSearchTool — Structured Google Search

Get structured Google Search results:

```python
from langchain_scraperapi.tools import ScraperAPIGoogleSearchTool

google_search = ScraperAPIGoogleSearchTool()

results = google_search.invoke({
    "query": "what is langchain",
    "num": 20,
    "output_format": "json"
})
print(results)
```

**Parameters:**
- `query` (required) – search terms
- `output_format` – `"json"` (default) or `"csv"`
- `country_code`, `tld`, `num`, `hl`, `gl` – optional search modifiers

### ScraperAPIAmazonSearchTool — Structured Amazon Search

Get structured Amazon product search results:

```python
from langchain_scraperapi.tools import ScraperAPIAmazonSearchTool

amazon_search = ScraperAPIAmazonSearchTool()

products = amazon_search.invoke({
    "query": "noise cancelling headphones",
    "tld": "co.uk",
    "page": 2
})
print(products)
```

**Parameters:**
- `query` (required) – product search terms
- `output_format` – `"json"` (default) or `"csv"`
- `country_code`, `tld`, `page` – optional search modifiers

## Example: AI Agent that can browse the web

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_scraperapi.tools import ScraperAPITool

# Set up tools and LLM
tools = [ScraperAPITool()]
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can browse websites. Use ScraperAPITool to access web content."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create and run agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response = agent_executor.invoke({
    "input": "Browse hackernews and summarize the top story"
})
```

## Documentation

For complete parameter details and advanced usage, see the [ScraperAPI documentation](https://docs.scraperapi.com/python/making-requests/customizing-requests).