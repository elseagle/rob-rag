import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.chat_engine.types import ChatMode
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer

openai.api_key = st.secrets["OPENAI_API_KEY"]

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)


def reset_conversation():
    st.session_state["messages"] = None


prompt_template = PromptTemplate(
    """
YOU ARE A RETRIEVAL-AUGMENTED GENERATION AI TRAINED AND DEVELOPED BY INSOMNIA LABS. 
Given the following {query_str}

Your response should be:

1. Your response should prioritize clarity, conciseness, and relevance to the context of your knowledge base. 
3. Clear, concise, and relevant, avoiding unnecessary elaboration unless requested.
4. Focused on the context and content within the Insomnia Labs knowledge base given to you.
5. Providing value through precision and relevance to enhance the user's experience.
"""
)
# page header
st.header("Maybelline Roblox Activation - FAQ & How-tos")
st.subheader("Ask me anything about the Maybelline Roblox Activation")
st.button("Reset Conversation", on_click=reset_conversation)


# initialize chat history
def initialize_chat():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm your Roblox assistant."}
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


def display_chat_messages(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def get_conversation_from_llm(index_param, user_prompt_param, memory=memory):
    chat_engine = index_param.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        memory=memory,
        llm=OpenAI(model="gpt-4-turbo-preview"),
        verbose=True,
    )

    return chat_engine.chat(prompt_template.format(query_str=user_prompt_param))


# Proceed only if the data is loaded successfully
if index is not None:
    print(st.session_state, end="\n\n")
    if st.session_state.get("messages") is None:
        initialize_chat()

    display_chat_messages(st.session_state["messages"])

    user_prompt = st.chat_input("Ask Away..")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        # Generate a new response if the last message is not from the assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = get_conversation_from_llm(index, user_prompt)
                        st.write(response.response)

                        # Add the assistant's response to the message history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response.response}
                        )
                    except Exception as e:
                        st.error(f"Failed to generate a response: {e}")
