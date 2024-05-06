
from llama_index.core.evaluation import generate_question_context_pairs, EmbeddingQAFinetuneDataset
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
    VectorStoreIndex,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
import os

# Load environment variables
load_dotenv()

# Fetch and set API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
active_loop_token = os.getenv("ACTIVELOOP_TOKEN")
dataset_path = os.getenv("DATASET_PATH")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
cohere_api_key = os.environ["COHERE_API_KEY"]


# create global settings for llm model and embedding model
Settings.llm = OpenAI(model="gpt-4-1106-preview")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# create a parser for the documents
parser = LlamaParse(result_type="markdown")
# create a file extractor mostly for pdf files
file_extractor = {".pdf": parser}

# load documents from a directory
documents = SimpleDirectoryReader("new", recursive=True, file_extractor=file_extractor).load_data()

# create the node spliter
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

# # creating in memory vector store
vector_store = VectorStoreIndex.from_documents(documents, show_progress=True)

# # save the vector store to local storage
# vector_store.storage_context.persist(persist_dir="vector_storage")

# creating re-ranking using cohere
cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=2)


# use the vector store from local storage
# storage_context = StorageContext.from_defaults(persist_dir="vector_storage")
# local_vector_store = load_index_from_storage(storage_context)
# # create a query engine
# query_engine = local_vector_store.as_query_engine(
#     streaming=True,
#     similarity_top_k=4,
#     node_postprocessor=[cohere_rerank],
#     )

# create query engine
query_engine = vector_store.as_query_engine(
    streaming=True,
    similarity_top_k=4,
    node_postprocessor=[cohere_rerank],)
# create streaming response
# while (query := input("Enter your query: ")) != "exit":
#     response = query_engine.query(query)
#     response.print_response_stream()
# without using node splitter
# index = VectorStoreIndex.from_documents(
#     documents,
#     show_progress=True,
# )
# query_engine = index.as_query_engine(streaming=True, similarity_top_k=4)
# create streaming response
# streaming_response = query_engine.query("what are the adoption rates of electric vehicles in the hong kong, and give full explanation?")
# streaming_response.print_response_stream()


    

# # creating a sub query engine
# query_engine = vector_store.as_query_engine(similarity_top_k=4)

# # create a tool for the query engine
# query_engine_tools = [
#     QueryEngineTool(
#         query_engine=query_engine,
#         metadata=ToolMetadata(
#             name="ev_search",
#             description="A tool for searching information about electric vehicles",
#         ),
#     )
# ]

# create a sub question query engine
# sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
#     query_engine_tools=query_engine_tools,
#     use_async=True,
# )
# response = sub_question_query_engine.query("how can i charge my electric vehicle?, and which is the best charger to install?")
# print(">>> The final response:\n", response )

