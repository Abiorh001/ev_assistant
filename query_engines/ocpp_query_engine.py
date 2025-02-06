import logging
import os
import sys
from typing import Tuple, Union

from dotenv import load_dotenv
from llama_index.core import (ServiceContext, Settings, SimpleDirectoryReader,
                              StorageContext, VectorStoreIndex,
                              load_index_from_storage, get_response_synthesizer)
from llama_index.core.node_parser import SentenceSplitter, SimpleNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_parse import LlamaParse
from pinecone import Pinecone, PineconeApiException, ServerlessSpec

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load environment variables
load_dotenv()

Settings.llm = OpenAI(temperature=0.2, model="gpt-4-1106-preview")
class OCPPQueryEngine:
    def __init__(self):
        self.openai_api_key: Union[str, None] = os.getenv("OPENAI_API_KEY")
        self.llama_cloud_api_key: Union[str, None] = os.getenv("LLAMA_CLOUD_API_KEY")
        self.cohere_api_key: Union[str, None] = os.environ.get("COHERE_API_KEY")
        self.pinecone_api_key: Union[str, None] = os.environ.get("PINECONE_API_KEY")
        # self.LOCAL_STORAGE_PATH: str = "vector_storage"
        # self.llm: OpenAI = OpenAI(model="gpt-4-1106-preview", max_tokens=1000)
        self.embed_model: OpenAIEmbedding = OpenAIEmbedding(
            model="text-embedding-ada-002"
        )
        self.validate_api_keys()

    def validate_api_keys(self) -> None:
        if not all(
            [
                self.openai_api_key,
                self.llama_cloud_api_key,
                self.cohere_api_key,
                self.pinecone_api_key,
            ]
        ):
            raise ValueError(
                "One or more API keys are missing. List of api keys are: openai_api_key, llama_cloud_api_key, cohere_api_key"
            )

    def create_pinecone_index(self) -> VectorStoreIndex:
        pc = Pinecone(api_key=self.pinecone_api_key)

        index_name = "ocpp-index"
        try:
            pc.create_index(
                index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            pinecone_index = pc.Index(index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            documents = SimpleDirectoryReader(
                "data", recursive=True, file_extractor=None
            ).load_data()
            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
            nodes = splitter.get_nodes_from_documents(documents)
            vector_index = VectorStoreIndex(
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True,
                nodes=nodes,
            )
        except (PineconeApiException, Exception):
            pinecone_index = pc.Index(index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model=self.embed_model
            )

        return vector_index

    def create_load_local_vector_index(self) -> VectorStoreIndex:
        if not os.path.exists(self.LOCAL_STORAGE_PATH):
            # parser = LlamaParse(result_type="markdown")
            # file_extractor = {".pdf": parser}
            documents = SimpleDirectoryReader(
                "data", recursive=True, file_extractor=None
            ).load_data()
            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
            nodes = splitter.get_nodes_from_documents(documents)
            vector_store_index = VectorStoreIndex(
                nodes=nodes, embed_model=self.embed_model, show_progress=True
            )
            vector_store_index.storage_context.persist(
                persist_dir=self.LOCAL_STORAGE_PATH
            )
        else:
            storage_context = StorageContext.from_defaults(
                persist_dir=self.LOCAL_STORAGE_PATH
            )
            vector_store_index = load_index_from_storage(storage_context)
        return vector_store_index

    def create_BM25_and_vector_retriever(
        self,
    ) -> Tuple[BM25Retriever, VectorIndexRetriever]:
        # load or create the pinecone vector store index
        pinecone_vector_index = self.create_pinecone_index()
        bm25_retriever = BM25Retriever.from_defaults(
            index=pinecone_vector_index, similarity_top_k=8
        )
        vector_index_retriever = VectorIndexRetriever(
            pinecone_vector_index, similarity_top_k=8
        )
        return bm25_retriever, vector_index_retriever


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_index_retriever, bm25_retriever):
        self.vector_index_retriever = vector_index_retriever
        self.bm25_retriever = bm25_retriever

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_index_retriever.retrieve(query, **kwargs)
        all_nodes = []
        nodes_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in nodes_ids:
                all_nodes.append(n)
                nodes_ids.add(n.node.node_id)
        return all_nodes


ocpp_query = OCPPQueryEngine()
# bm25_retriever, vector_index_retriever = ocpp_query.create_BM25_and_vector_retriever()
# hybrid_retriever = HybridRetriever(vector_index_retriever, bm25_retriever)
vector_index = ocpp_query.create_pinecone_index()
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=8)
# documents = SimpleDirectoryReader(
#                 "data", recursive=True, file_extractor=None
#             ).load_data()
# splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
# nodes = splitter.get_nodes_from_documents(documents)
# vector_retriever = VectorStoreIndex(
#                 nodes=nodes, embed_model=ocpp_query.embed_model, show_progress=True
#             )
cohere_rerank = CohereRerank(api_key=ocpp_query.cohere_api_key, top_n=8)
# configure response synthesizer
response_synthesizer = get_response_synthesizer()
ocpp_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[cohere_rerank]
)

# response =  ocpp_query_engine.query("in the document explain to me the oca whitepapers")
# print(response)
