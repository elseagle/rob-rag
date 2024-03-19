import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine.types import ChatMode
import openai
from llama_index.llms.openai import OpenAI

# Securely load the OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Setting up the page header
st.header("Insomnia Labs - Roblox")
st.subheader("Ask a question about Roblox and get an answer from my knowledge base")

# Initialize the chat message history
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hello! I'm your assistant. Ask me anything about Roblox."},
    ]

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
    chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONDENSE_QUESTION, llm=OpenAI(model="gpt-4-turbo-preview"), verbose=True)

    if prompt := st.text_input("Ask me anything about Roblox"):
        # Add the user's prompt to the message history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Generate a new response if the last message is not from the assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = chat_engine.chat(prompt)
                        st.write(response.response)
                        # Add the assistant's response to the message history
                        st.session_state.messages.append({"role": "assistant", "content": response.response})
                    except Exception as e:
                        st.error(f"Failed to generate a response: {e}")
