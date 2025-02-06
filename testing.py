from llama_index.llms.openai import OpenAI
import asyncio
from typing import Any, List, Callable, Optional, Union, Dict
from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn,
)
import re
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import PropertyGraphIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.graph_stores import SimplePropertyGraphStore
import networkx as nx
from graspologic.partition import hierarchical_leiden
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

class GraphRAGStore(SimplePropertyGraphStore):
    community_summary = {}
    max_cluster_size = 5

    def generate_community_summary(self, text):
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Analyze these knowledge graph relationships and create a comprehensive summary. "
                    "Include: key entities, their relationships, and overarching themes. "
                    "Highlight any patterns or significant connections. "
                    "Format with sections:\n"
                    "1. Main Themes\n2. Key Entities\n3. Significant Relationships\n4. Narrative Overview"
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = llm.chat(messages)
        return self._clean_response(response)

    def _create_nx_graph(self):
        nx_graph = nx.Graph()
        # Add nodes with original IDs
        for node_id in self.graph.nodes:
            nx_graph.add_node(node_id)
        # Add edges with attributes
        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        community_info = {}

        # Handle clustered nodes
        for item in clusters:
            cluster_id = item.cluster
            if cluster_id not in community_info:
                community_info[cluster_id] = []
            
            for neighbor in nx_graph.neighbors(item.node):
                if any(n.cluster == cluster_id for n in clusters if n.node == neighbor):
                    edge_data = nx_graph.get_edge_data(item.node, neighbor)
                    node_data = self.graph.nodes.get(item.node, None)
                    neighbor_data = self.graph.nodes.get(neighbor, None)

                    if node_data and neighbor_data and isinstance(node_data, EntityNode) and isinstance(neighbor_data, EntityNode):
                        detail = (
                            f"{node_data.name} -> {neighbor_data.name} -> "
                            f"{edge_data['relationship']} -> {edge_data['description']}"
                        )
                        community_info[cluster_id].append(detail)

        # Handle isolated nodes safely
        isolates = list(nx.isolates(nx_graph))
        for node_id in isolates:
            node_data = self.graph.nodes.get(node_id, None)
            if node_data and isinstance(node_data, EntityNode):
                community_info[node_id] = [
                    f"Isolated node: {node_data.name} - {node_data.properties.get('description', '')}"
                ]
        
        return community_info


    def _summarize_communities(self, community_info):
        self.community_summary.clear()
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."
            self.community_summary[community_id] = self.generate_community_summary(details_text)

    def build_communities(self):
        nx_graph = self._create_nx_graph()
        non_isolates = [n for n in nx_graph.nodes if nx_graph.degree(n) > 0]
        
        if non_isolates:
            subgraph = nx_graph.subgraph(non_isolates)
            clusters = hierarchical_leiden(subgraph, max_cluster_size=self.max_cluster_size)
        else:
            clusters = []
        
        community_info = self._collect_community_info(nx_graph, clusters)
        self._summarize_communities(community_info)

    def _clean_response(self, response):
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM

    def custom_query(self, query_str: str) -> str:
        self.graph_store.build_communities()
        community_summaries = self.graph_store.community_summary
        
        answers = []
        for comm_id, summary in community_summaries.items():
            answer = self._generate_answer(summary, query_str)
            answers.append(answer)
        
        return self._aggregate_answers(answers)

    def _generate_answer(self, summary, query):
        prompt = f"""Community Context: {summary}
        
        Query: {query}
        
        Instructions:
        1. Directly answer using ONLY the provided context
        2. If irrelevant, state "Not covered in this context"
        3. Keep responses factual and concise"""
        
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content="Generate response:"),
        ]
        return self._clean_response(self.llm.chat(messages))

    def _aggregate_answers(self, answers):
        prompt = f"""Synthesize these partial answers into a final response:
        
        {answers}
        
        Structure your answer with:
        1. Key Themes
        2. Main Entities
        3. Critical Relationships
        4. Comprehensive Narrative
        
        Remove redundancies while preserving important nuances."""
        
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", content="Generate final answer:"),
        ]
        return self._clean_response(self.llm.chat(messages))

    def _clean_response(self, response):
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$"<entity_name>"$$$$"<entity_type>"$$$$"<entity_description>")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$"<source_entity>"$$$$"<target_entity>"$$$$"<relation>"$$$$"<relationship_description>")

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:"""

# Then continue with the parse_fn definition
entity_pattern = r'\("entity"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"((?:[^"]|\\")*)"\s*\)'
relationship_pattern = r'\("relationship"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"([^"]+)"\s*\$\$\$\$\s*"((?:[^"]|\\")*)"\s*\)'
def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships

# Implementation
documents = SimpleDirectoryReader("ragtest", recursive=True).load_data()
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=10,  # Increased for better coverage
    parse_fn=parse_fn,
)

index = PropertyGraphIndex(
    nodes=nodes,
    property_graph_store=GraphRAGStore(),
    kg_extractors=[kg_extractor],
    show_progress=True,
)

# Example usage
print(f"Total nodes processed: {len(nodes)}")
print(f"Graph contains {len(index.property_graph_store.graph.nodes)} entities")
print(f"Graph contains {len(index.property_graph_store.graph.relations)} relationships")

index.property_graph_store.build_communities()
print(f"Detected {len(index.property_graph_store.community_summary)} communities")

query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store, 
    llm=llm
)

response = query_engine.query(
    "Analyze the document's key themes and character relationships. "
    "Structure your answer with sections for: "
    "1. Main Themes 2. Character Dynamics 3. Historical Context"
)
print("\nFinal Response:")
print(response)