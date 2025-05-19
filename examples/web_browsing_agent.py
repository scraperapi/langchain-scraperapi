import os
import dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_scraperapi.tools import ScraperAPITool

dotenv.load_dotenv()


scraper_tool = ScraperAPITool(output_format="markdown", premium=True, render=True)
tools = [scraper_tool]
llm = ChatOpenAI(model_name="gpt-4.1", temperature=0)

st.set_page_config(page_title="🌐 Web Browse Agent", page_icon="🤖")

st.markdown("""
<style>
img {
    max-width: 500px !important;
    height: auto;
}
</style>
""", unsafe_allow_html=True)

st.title("🌐 Web Browse Agent")
st.caption("I can browse websites for you! Just give me a URL and what you're looking for.")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can browse websites for users. When asked to browse a website, use the ScraperAPITool. You will be given the URL. Always try to get the content in markdown format if possible. If the user asks a question about a website that requires Browse, infer the URL if not explicitly given or ask the user for the URL."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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
                st.markdown(str(observation))

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! How can I help you browse the web today?")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)
    if msg.type == "ai" and "intermediate_steps" in msg.additional_kwargs:
         render_tool_expander(msg.additional_kwargs["intermediate_steps"])


if user_query := st.chat_input("Ask me to browse a website... (e.g., 'Browse example.com and tell me about their plans')"):
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
