from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
import textwrap
import openai
import os
from postgres_vector_db.pgvector_store import vector_store

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.environ["OPENAI_API_KEY"]

#documents = SimpleDirectoryReader("data", recursive=True).load_data()


storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    storage_context=storage_context, show_progress=True
)
query_engine = index.as_query_engine()

response = query_engine.query("what is the documents all about?")
print(textwrap.fill(str(response), 100))