from llama_index.llms.openai import OpenAI
import asyncio
from typing import Any, List, Callable, Optional, Union, Dict
from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,
)
from collections import defaultdict
import re
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.graph_stores import SimplePropertyGraphStore
import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

from llama_index.core.llms import ChatMessage

from llama_index.core.graph_stores.types import (
    EntityNode,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
)
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core import Settings, SimpleDirectoryReader


from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(model="gpt-4o")
Settings.llm = OpenAI(temperature=0.2, model="gpt-4-1106-preview")
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Async semaphore to rate-limit LLM calls (Adjust as needed)
SEMAPHORE = asyncio.Semaphore(5)  # Allows up to 5 concurrent requests

class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        async with SEMAPHORE:
            try:
                llm_response = await self.llm.apredict(
                    self.extract_prompt,
                    text=text,
                    max_knowledge_triplets=self.max_paths_per_chunk,
                )
                entities, entities_relationship = self.parse_fn(llm_response)
            except ValueError:
                entities = []
                entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )


class GraphRAGStore(Neo4jPropertyGraphStore):
    """
    Enhanced GraphRAGStore for Knowledge Graph-based Retrieval.
    This class builds communities (clusters) from triplets stored in the graph,
    then summarizes them using an LLM.
    """

    community_summary: Dict[int, str] = {}  # Maps community/cluster IDs to generated summaries.
    entity_info: Dict[str, List[int]] = {}  # Maps node identifiers to a list of community IDs.
    max_cluster_size: int = 5  # Maximum number of entities per cluster.

    async def generate_community_summary(self, text: str) -> str:
        """
        Generate a concise summary of community relationships using an LLM.
        
        Args:
            text (str): Concatenated relationship details from a community.
        
        Returns:
            str: The generated summary.
        """
        async with SEMAPHORE:
            try:
                messages = [
                    ChatMessage(
                        role="system",
                        content=(
                            "You are provided with knowledge graph relationships in the format: "
                            "entity1 -> entity2 -> relation -> relationship_description.\n"
                            "Your task: Generate a structured summary that captures key entities and significant relationships."
                        ),
                    ),
                    ChatMessage(role="user", content=text),
                ]
                response = await llm.achat(messages)
                return self._clean_response(response)
            except Exception as e:
                logger.error(f"LLM failed to generate summary: {e}")
                return "Summary generation failed."

    def build_communities(self) -> None:
        """
        Extracts communities from the knowledge graph by:
          1. Converting stored triplets into a NetworkX graph.
          2. Running hierarchical clustering (Leiden algorithm) on the graph.
          3. Collecting relationship details per community.
          4. Generating summaries asynchronously.
        """
        nx_graph = self._create_nx_graph()
        clusters = hierarchical_leiden(nx_graph, max_cluster_size=self.max_cluster_size)

        # _collect_community_info returns:
        # - entity_info: mapping from node_id to list of cluster IDs
        # - community_info: mapping from cluster ID to list of relationship details
        self.entity_info, community_info = self._collect_community_info(nx_graph, clusters)
        # Run async summarization. If already in an event loop, consider using create_task/await instead.
        asyncio.run(self._summarize_communities(community_info))

    def _create_nx_graph(self) -> nx.Graph:
        """
        Converts stored triplets into a NetworkX graph.
        
        Returns:
            nx.Graph: A graph where nodes represent entities and edges represent relationships.
        """
        nx_graph = nx.Graph()

        try:
            triplets = self.get_triplets()
            for entity1, relation, entity2 in triplets:
                # Use provided attributes or fall back to hash-based unique identifiers.
                entity1_id = getattr(entity1, "id", None) or getattr(entity1, "name", None) or f"Entity_{hash(entity1)}"
                entity2_id = getattr(entity2, "id", None) or getattr(entity2, "name", None) or f"Entity_{hash(entity2)}"

                # Add nodes with a "name" attribute.
                nx_graph.add_node(entity1_id, name=getattr(entity1, "name", str(entity1)))
                nx_graph.add_node(entity2_id, name=getattr(entity2, "name", str(entity2)))

                # Add an edge with relationship metadata.
                nx_graph.add_edge(
                    entity1_id,
                    entity2_id,
                    relationship=getattr(relation, "label", "Unknown"),
                    description=getattr(relation, "properties", {}).get("relationship_description", "N/A"),
                )
        except Exception as e:
            logger.error(f"Failed to create NetworkX graph: {e}")

        return nx_graph

    def _collect_community_info(self, nx_graph: nx.Graph, clusters: List[Any]) -> (Dict[str, List[int]], Dict[int, List[str]]):
        """
        Aggregates nodes into communities and collects relationship details.
        
        Args:
            nx_graph (nx.Graph): The NetworkX graph of entities.
            clusters (list): Clustering results with attributes 'node' and 'cluster'.
        
        Returns:
            tuple:
                - entity_info (dict): Mapping from node identifier to a list of community IDs.
                - community_info (dict): Mapping from community ID to a list of relationship details.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node_id = item.node
            cluster_id = item.cluster

            entity_info[node_id].add(cluster_id)

            for neighbor in nx_graph.neighbors(node_id):
                edge_data = nx_graph.get_edge_data(node_id, neighbor)
                if edge_data:
                    detail = (
                        f"{nx_graph.nodes[node_id].get('name', node_id)} -> "
                        f"{nx_graph.nodes[neighbor].get('name', neighbor)} -> "
                        f"{edge_data.get('relationship', 'Unknown')} -> "
                        f"{edge_data.get('description', 'No description')}"
                    )
                    community_info[cluster_id].append(detail)

        # Convert sets to lists for entity_info.
        entity_info = {node: list(clusters) for node, clusters in entity_info.items()}
        return entity_info, dict(community_info)

    async def _summarize_communities(self, community_info: Dict[int, List[str]]) -> None:
        """
        Asynchronously generates community summaries.
        
        Args:
            community_info (dict): Mapping from community ID to relationship details.
        """
        self.community_summary.clear()
        tasks = {
            community_id: self.generate_community_summary("\n".join(details))
            for community_id, details in community_info.items()
        }

        summaries = await asyncio.gather(*tasks.values())
        for idx, community_id in enumerate(tasks.keys()):
            self.community_summary[community_id] = summaries[idx]

    def get_community_summaries(self) -> Dict[int, str]:
        """
        Returns the community summaries, building them if they are not already generated.
        
        Returns:
            dict: Mapping from community ID to its summary.
        """
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

    @staticmethod
    def _clean_response(response: str) -> str:
        """
        Removes any unwanted assistant prefixes from LLM responses.
        
        Args:
            response (str): Raw response from the LLM.
        
        Returns:
            str: Cleaned response.
        """
        return re.sub(r"^assistant:\s*", "", str(response)).strip()


class GraphRAGQueryEngine(CustomQueryEngine):
    """
    Query engine that uses community summaries generated from a property graph.
    It finds relevant entities, retrieves their associated communities, and then uses the summaries
    to generate an answer via an LLM.
    """

    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: Any  # Replace with the appropriate LLM type if available.
    similarity_top_k: int = 20

    def custom_query(self, query_str: str) -> str:
        """
        Process all community summaries to generate an answer for a given query.
        
        Args:
            query_str (str): The user query.
        
        Returns:
            str: The final aggregated answer.
        """
        # Retrieve entities that match the query.
        entities = self.get_entities(query_str, self.similarity_top_k)

        # Find community/cluster IDs associated with these entities.
        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        # Get the latest summaries (build if not present).
        community_summaries = self.graph_store.get_community_summaries()

        # Generate answers for each relevant community.
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for community_id, community_summary in community_summaries.items()
            if community_id in community_ids
        ]

        # Aggregate all community answers into a final answer.
        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def get_entities(self, query_str: str, similarity_top_k: int) -> List[str]:
        """
        Retrieve entities from the index based on the query.
        
        Args:
            query_str (str): The user query.
            similarity_top_k (int): Number of top similar results to retrieve.
        
        Returns:
            list: A list of entity names.
        """
        nodes_retrieved = self.index.as_retriever(similarity_top_k=similarity_top_k).retrieve(query_str)
        entities = set()
        # regex to capture two entities separated by arrows.
        pattern = r"^(\w+(?:\s+\w+)*)\s*->\s*[a-zA-Z\s]+?\s*->\s*(\w+(?:\s+\w+)*)$"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                subject, obj = match
                entities.add(subject)
                entities.add(obj)
        return list(entities)

    def retrieve_entity_communities(self, entity_info: Dict[str, List[int]], entities: List[str]) -> List[int]:
        """
        Retrieve community IDs for given entities, allowing for multiple communities per entity.
        
        Args:
            entity_info (dict): Mapping from node identifier to list of community IDs.
            entities (list): List of entity names.
        
        Returns:
            list: Unique list of community IDs that the entities belong to.
        """
        community_ids = []
        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])
        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary: str, query: str) -> str:
        """
        Generate an answer based on a community summary and the query.
        
        Args:
            community_summary (str): The summary of a community.
            query (str): The original query.
        
        Returns:
            str: The generated answer.
        """
        prompt = (
            f"Given the community summary:\n{community_summary}\n\n"
            f"How would you answer the following query?\nQuery: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content="I need an answer based on the above information."),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers: List[str]) -> str:
        """
        Aggregate individual community answers into a final, coherent response.
        
        Args:
            community_answers (list): Answers generated from different community summaries.
        
        Returns:
            str: The final aggregated answer.
        """
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content=f"Intermediate answers: {community_answers}"),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(r"^assistant:\s*", "", str(final_response)).strip()
        return cleaned_final_response



documents = SimpleDirectoryReader("Data", recursive=True, file_extractor=None).load_data()
splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)
nodes = splitter.get_nodes_from_documents(documents)
KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities with their corresponding types and detailed descriptions, and then determine all clearly related pairs among those entities. From the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. **Entity Extraction**  
   Identify all entities present in the text. For each identified entity, extract the following information:
   - **entity_name**: The capitalized name of the entity.
   - **entity_type**: The type or category of the entity.
   - **entity_description**: A comprehensive description of the entity's attributes, roles, and activities.

   **Format each entity as:**  
   `("entity_name"$$$$"entity_type"$$$$"entity_description")`

2. **Relationship Extraction**  
   From the entities identified in step 1, determine all pairs of entities that are *clearly related*. For each pair, extract:
   - **source_entity**: The name of the source entity (as identified in step 1).
   - **target_entity**: The name of the target entity (as identified in step 1).
   - **relation**: The relationship between the source and target entities.
   - **relationship_description**: An explanation detailing why these two entities are considered related.

   **Format each relationship as:**  
   `("source_entity"$$$$"target_entity"$$$$"relation"$$$$"relationship_description")`

3. **Output Format**  
   Once the entities and relationships are extracted, output all triplets (entity and relationship pairs) using the above formats.

-Real Data-
######################
text: {text}
######################
output:
"""

entity_pattern = r'\("((?:\\.|[^"\\])*)"\$\$\$\$"((?:\\.|[^"\\])*)"\$\$\$\$"((?:\\.|[^"\\])*)"\)'

relationship_pattern = r'\("((?:\\.|[^"\\])*)"\$\$\$\$"((?:\\.|[^"\\])*)"\$\$\$\$"((?:\\.|[^"\\])*)"\$\$\$\$"((?:\\.|[^"\\])*)"\)'

def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships


kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=10,
    parse_fn=parse_fn,
)

graph_store = GraphRAGStore(
    username="neo4j", password="Lucifer_001", url="bolt://localhost:7687"
)
index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    show_progress=True,
)
print(index.property_graph_store.get_triplets()[10])
print(index.property_graph_store.get_triplets()[10][0].properties)
print(index.property_graph_store.get_triplets()[10][1].properties)
index.property_graph_store.build_communities()
query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store,
    llm=llm,
    index=index,
    similarity_top_k=10,
)


response = query_engine.query(
    "What are the main thing discussed in the document?"
)
print(response)

response = query_engine.query(
    "Tell me about the relationship between the entities in the document."
)
print(response)


# this is use when no neo4j is available
# class GraphRAGStore(SimplePropertyGraphStore):
#     """This class is used to generate the communities from the nodes and relations in the graph and summarize them."""
#     community_summary = {}
#     max_cluster_size = 10

#     def generate_community_summary(self, text):
#         """Generate summary for a given text using an LLM."""
#         messages = [
#             ChatMessage(
#                 role="system",
#                 content=(
#                     "You are provided with a set of relationships from a knowledge graph, each represented as "
#                     "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
#                     "relationships. The summary should include the names of the entities involved and a concise synthesis "
#                     "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
#                     "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
#                     "integrates the information in a way that emphasizes the key aspects of the relationships."
#                 ),
#             ),
#             ChatMessage(role="user", content=text),
#         ]
#         response = llm.chat(messages)
#         clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
#         return clean_response
    
#     def _create_nx_graph(self):
#         """Converts internal graph representation to NetworkX graph."""
#         nx_graph = nx.Graph()
#         for node in self.graph.nodes.values():
#             nx_graph.add_node(str(node))
#         for relation in self.graph.relations.values():
#             nx_graph.add_edge(
#                 relation.source_id,
#                 relation.target_id,
#                 relationship=relation.label,
#                 description=relation.properties["relationship_description"],
#             )
#         return nx_graph
    
#     def _collect_community_info(self, nx_graph, clusters):
#         """Collect detailed information for each node based on their community."""
#         community_mapping = {item.node: item.cluster for item in clusters}
#         community_info = {}
#         for item in clusters:
#             cluster_id = item.cluster
#             node = item.node
#             if cluster_id not in community_info:
#                 community_info[cluster_id] = []

#             for neighbor in nx_graph.neighbors(node):
#                 if community_mapping[neighbor] == cluster_id:
#                     edge_data = nx_graph.get_edge_data(node, neighbor)
#                     if edge_data:
#                         detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
#                         community_info[cluster_id].append(detail)
#         return community_info
    
#     def _summarize_communities(self, community_info):
#         """Generate and store summaries for each community."""
#         for community_id, details in community_info.items():
#             details_text = (
#                 "\n".join(details) + "."
#             )  # Ensure it ends with a period
#             self.community_summary[
#                 community_id
#             ] = self.generate_community_summary(details_text)
            
            
#     def build_communities(self):
#         """Builds communities from the graph and summarizes them."""
#         nx_graph = self._create_nx_graph()
#         community_hierarchical_clusters = hierarchical_leiden(
#             nx_graph, max_cluster_size=self.max_cluster_size
#         )
#         community_info = self._collect_community_info(
#             nx_graph, community_hierarchical_clusters
#         )
#         self._summarize_communities(community_info)


#     def get_community_summaries(self):
#         """Returns the community summaries, building them if not already done."""
#         if not self.community_summary:
#             self.build_communities()
#         return self.community_summary






# class GraphRAGQueryEngine(CustomQueryEngine):
#     graph_store: GraphRAGStore
#     llm: LLM

#     def custom_query(self, query_str: str) -> str:
#         """Process all community summaries to generate answers to a specific query."""
#         community_summaries = self.graph_store.get_community_summaries()
#         community_answers = [
#             self.generate_answer_from_summary(community_summary, query_str)
#             for _, community_summary in community_summaries.items()
#         ]

#         final_answer = self.aggregate_answers(community_answers)
#         return final_answer

#     def generate_answer_from_summary(self, community_summary, query):
#         """Generate an answer from a community summary based on a given query using LLM."""
#         prompt = (
#             f"Given the community summary: {community_summary}, "
#             f"how would you answer the following query? Query: {query}"
#         )
#         messages = [
#             ChatMessage(role="system", content=prompt),
#             ChatMessage(
#                 role="user",
#                 content="I need an answer based on the above information.",
#             ),
#         ]
#         response = self.llm.chat(messages)
#         cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
#         return cleaned_response

#     def aggregate_answers(self, community_answers):
#         """Aggregate individual community answers into a final, coherent response."""
#         # intermediate_text = " ".join(community_answers)
#         prompt = "Combine the following intermediate answers into a final, concise response."
#         messages = [
#             ChatMessage(role="system", content=prompt),
#             ChatMessage(
#                 role="user",
#                 content=f"Intermediate answers: {community_answers}",
#             ),
#         ]
#         final_response = self.llm.chat(messages)
#         cleaned_final_response = re.sub(
#             r"^assistant:\s*", "", str(final_response)
#         ).strip()
#         return cleaned_final_response
    

# documents = SimpleDirectoryReader("ragtest", recursive=True, file_extractor=None).load_data()
# splitter = SentenceSplitter(
#     chunk_size=1024,
#     chunk_overlap=20,
# )
# nodes = splitter.get_nodes_from_documents(documents)


# KG_TRIPLET_EXTRACT_TMPL = """
# -Goal-
# Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
# Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

# -Steps-
# 1. Identify all entities. For each identified entity, extract the following information:
# - entity_name: Name of the entity, capitalized
# - entity_type: Type of the entity
# - entity_description: Comprehensive description of the entity's attributes and activities
# Format each entity as ("entity"$$$$"<entity_name>"$$$$"<entity_type>"$$$$"<entity_description>")

# 2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
# For each pair of related entities, extract the following information:
# - source_entity: name of the source entity, as identified in step 1
# - target_entity: name of the target entity, as identified in step 1
# - relation: relationship between source_entity and target_entity
# - relationship_description: explanation as to why you think the source entity and the target entity are related to each other

# Format each relationship as ("relationship"$$$$"<source_entity>"$$$$"<target_entity>"$$$$"<relation>"$$$$"<relationship_description>")

# 3. When finished, output.

# -Real Data-
# ######################
# text: {text}
# ######################
# output:"""

# entity_pattern = r'\("entity"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"((?:[^"]|\\")*)"\s*\)'
# relationship_pattern = r'\("relationship"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"((?:[^"]|\\")*)"\s*\)'

# def parse_fn(response_str: str) -> Any:
#     entities = re.findall(entity_pattern, response_str)
#     relationships = re.findall(relationship_pattern, response_str)
#     return entities, relationships


# kg_extractor = GraphRAGExtractor(
#     llm=llm,
#     extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
#     max_paths_per_chunk=10,
#     parse_fn=parse_fn,
# )

# index = PropertyGraphIndex(
#     nodes=nodes,
#     property_graph_store=GraphRAGStore(),
#     kg_extractors=[kg_extractor],
#     show_progress=True,
# )
# print(len(nodes))
# print(list(index.property_graph_store.graph.nodes.values())[-1])
# print(list(index.property_graph_store.graph.relations.values())[0])
# print(list(index.property_graph_store.graph.relations.values())[0].properties[
#     "relationship_description"
# ])
# index.property_graph_store.build_communities()
# query_engine = GraphRAGQueryEngine(
#     graph_store=index.property_graph_store, llm=llm
# )
# response = query_engine.query(
#     "What are the main thing discussed in the document?"
# )
# print(response)