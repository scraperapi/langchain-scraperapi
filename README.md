# langchain-scraperapi

This package contains the LangChain integration with ScraperAPI

## Installation

```bash
pip install -U langchain-scraperapi
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatScraperAPI` class exposes chat models from ScraperAPI.

```python
from langchain_scraperapi import ChatScraperAPI

llm = ChatScraperAPI()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`ScraperAPIEmbeddings` class exposes embeddings from ScraperAPI.

```python
from langchain_scraperapi import ScraperAPIEmbeddings

embeddings = ScraperAPIEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`ScraperAPILLM` class exposes LLMs from ScraperAPI.

```python
from langchain_scraperapi import ScraperAPILLM

llm = ScraperAPILLM()
llm.invoke("The meaning of life is")
```
