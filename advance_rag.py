import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import (ServiceContext, Settings, SimpleDirectoryReader,
                              StorageContext, VectorStoreIndex,
                              load_index_from_storage)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_parse import LlamaParse

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load environment variables
load_dotenv()

# Fetch and set API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
cohere_api_key = os.environ["COHERE_API_KEY"]


# create global settings for llm model and embedding model
Settings.llm = OpenAI(model="gpt-4-1106-preview")
llm = Settings.llm
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# initialized a global local storage path
LOCAL_STORAGE_PATH = "vector_storage"
# check if the local storage path exists and create it if it does not
if not os.path.exists(LOCAL_STORAGE_PATH):
    # create a parser for the documents
    parser = LlamaParse(result_type="markdown")
    # create a file extractor mostly for pdf files
    file_extractor = {".pdf": parser}

    # load documents from a directory
    documents = SimpleDirectoryReader(
        "new",
        recursive=True,
        file_extractor=file_extractor).load_data()

    # create the node parser using sentence splitter
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)

    # initialize the vector store using sentence splitter node parser
    vector_store_index = VectorStoreIndex(nodes, show_progress=True)
    # save the vector store to local storage
    vector_store_index.storage_context.persist(persist_dir=LOCAL_STORAGE_PATH)
    # use the vector store from local storage
    storage_context = StorageContext.from_defaults(persist_dir=LOCAL_STORAGE_PATH)
    local_vector_index = load_index_from_storage(storage_context)
else:
    # use the vector store from local storage
    storage_context = StorageContext.from_defaults(persist_dir=LOCAL_STORAGE_PATH)
    local_vector_index = load_index_from_storage(storage_context)


# create a bm25 retriever
bm25_retriever = BM25Retriever.from_defaults(
    index=local_vector_index,
    similarity_top_k=8
)

# create a vector index retriever
vector_index_retriever = VectorIndexRetriever(
    local_vector_index,
    similarity_top_k=8
)


# create a custom base hybrid retriever
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_index_retriever, bm25_retriever):
        self.vector_index_retriever = vector_index_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_index_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        nodes_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in nodes_ids:
                all_nodes.append(n)
                nodes_ids.add(n.node.node_id)
        return all_nodes


# create a hybrid retriever
hybrid_retriever = HybridRetriever(vector_index_retriever, bm25_retriever)

# creating re-ranking using cohere
cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=8)

# create a query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[cohere_rerank],
    #streaming=True,
    llm=Settings.llm,
)

# while (query := input("Enter your query: ")) != "exit":
#     response = query_engine.query(query)
#     response.print_response_stream()