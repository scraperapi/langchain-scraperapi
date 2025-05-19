import os
import dotenv
import streamlit as st
import json

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_scraperapi.tools import ScraperAPIAmazonSearchTool

dotenv.load_dotenv()

amazon_search_tool = ScraperAPIAmazonSearchTool()

tools = [amazon_search_tool]

llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         f"You are a helpful assistant that searches for products on Amazon. "
         f"When asked to search Amazon, use the '{amazon_search_tool.name}' tool. "
         "The tool requires a 'query' parameter for the search term. "
         "It can also optionally take a 'page' parameter for the page number (defaults to 1). "
         "Please extract the search query and page number (if the user specifies one) from the user's request."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.set_page_config(page_title="📦 Amazon Search Agent", page_icon="🛒")

st.markdown("""
<style>
img {
    max-width: 500px !important;
    height: auto;
}
</style>
""", unsafe_allow_html=True)

st.title("📦 Amazon Search Agent")
st.caption("I can search Amazon for you! Just tell me what you're looking for.")

def render_tool_expander(intermediate_steps):
    if intermediate_steps:
        with st.expander("🔎 Tool Interaction Details", expanded=False):
            for step in intermediate_steps:
                action = step[0]
                observation = step[1]

                st.markdown("**Tool Called:**")
                st.text(action.tool)
                st.markdown("**Tool Input:**")
                if isinstance(action.tool_input, dict):
                    st.json(action.tool_input)
                else:
                    st.text(action.tool_input)

                st.markdown("**Tool Output (Observation):**")
                try:
                    if isinstance(observation, (dict, list)):
                        st.json(observation)
                    elif isinstance(observation, str):
                         parsed_json = json.loads(observation)
                         st.json(parsed_json)
                    else:
                         st.text(str(observation))
                except json.JSONDecodeError:
                    st.text(str(observation))
                except Exception:
                    st.text(str(observation))


msgs = StreamlitChatMessageHistory(key="amazon_search_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! How can I help you search Amazon today?")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
    if msg.type == "ai" and "intermediate_steps" in msg.additional_kwargs:
        render_tool_expander(msg.additional_kwargs["intermediate_steps"])


if user_query := st.chat_input("What should I search for on Amazon?"):
    msgs.add_user_message(user_query)
    st.chat_message("user").write(user_query)

    agent_input = {"input": user_query, "chat_history": msgs.messages[:-1]}

    with st.chat_message("ai"):
        final_response_content = ""
        intermediate_steps = []

        try:
            with st.spinner("Agent is working..."):
                agent_executor_with_steps = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    return_intermediate_steps=True
                )
                response_with_steps = agent_executor_with_steps.invoke(agent_input)
                final_response_content = response_with_steps.get("output", "Sorry, I couldn't process that request.")
                intermediate_steps = response_with_steps.get("intermediate_steps", [])

            render_tool_expander(intermediate_steps)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            final_response_content = "I encountered an error trying to process your request."
            with st.expander("🔎 Tool Interaction Details", expanded=True):
                st.error(f"Error during agent execution: {e}")

        st.write(final_response_content)

        ai_message_to_add = AIMessage(
            content=final_response_content,
            additional_kwargs={"intermediate_steps": intermediate_steps}
        )
        msgs.add_message(ai_message_to_add)