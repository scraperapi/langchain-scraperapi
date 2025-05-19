import dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_scraperapi.tools import ScraperAPIAmazonSearchTool

dotenv.load_dotenv()

tools = [ScraperAPIAmazonSearchTool(output_format="json")]
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can browse Amazon for users. When asked to browse the website or look for a product, use the ScraperAPIAmazonSearchTool."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({
    "input": "can you give me the cheapest mother's day gifts"
})
print(response)