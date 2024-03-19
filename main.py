import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine

from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load data
documents = SimpleDirectoryReader("./data").load_data()
print(len(documents))
print()

# # Index
# index = VectorStoreIndex.from_documents(documents)
#
# # Query
# query_engine = index.as_query_engine(similarity_top_k=2)
# response = query_engine.query("Tell me about Maybelline and their association with Roblox")
# print(response)


index = VectorStoreIndex(documents)

# run query with HyDE query transform
query_str = ("Answer the following question like a Discord Mod: \n\n"
             "Given the recent news about Maybelline, what is their association with Roblox?")
hyde = HyDEQueryTransform(include_original=True)
query_engine = index.as_query_engine()
query_engine = TransformQueryEngine(query_engine, query_transform=hyde)
response = query_engine.query(query_str)
print(response)
