import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.chat_engine.types import ChatMode
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer

openai.api_key = st.secrets["OPENAI_API_KEY"]

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

prompt_template = PromptTemplate("""Given the following {query_str}

Your response should prioritize clarity, conciseness, and relevance to the context of your knowledge base. 
Avoid unnecessary elaboration and focus on delivering value to the user's experience within the event.
"""
                                 )
# page header
st.header("Maybelline Roblox Activation - FAQ & How-tos")
st.subheader("Ask me anything about the Maybelline Roblox Activation")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hello! I'm your Roblox assistant."},
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Function to load and index the data, with caching to avoid reloading on every interaction
@st.cache_resource(show_spinner=False)
def load_data():
    try:
        with st.spinner(text="Loading and indexing Roblox data..."):
            reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
            docs = reader.load_data()
            index = VectorStoreIndex.from_documents(docs)
        return index
    except Exception as e:
        st.error(f"Failed to load and index data: {e}")
        return None


index = load_data()

# Proceed only if the data is loaded successfully
if index is not None:
    chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT,
                                       memory=memory,
                                       llm=OpenAI(model="gpt-4-turbo-preview"),
                                       verbose=True)

    user_prompt = st.chat_input("Ask Away..")
    if user_prompt:
        # Add the user's prompt to the message history
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        # Display the user's prompt
        with st.chat_message("user"):
            st.write(user_prompt)

        # Generate a new response if the last message is not from the assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = chat_engine.chat(prompt_template.format(query_str=user_prompt))
                        st.write(response.response)
                        # Add the assistant's response to the message history
                        st.session_state.messages.append({"role": "assistant", "content": response.response})
                    except Exception as e:
                        st.error(f"Failed to generate a response: {e}")
