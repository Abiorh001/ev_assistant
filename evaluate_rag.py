from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    FaithfulnessEvaluator,
    BatchEvalRunner,
    RelevancyEvaluator
)
from advance_rag import nodes, Settings, hybrid_retriever, ServiceContext, OpenAI, VectorStoreIndex, query_engine
import pandas as pd


# create question context pairs
question_context_pairs = generate_question_context_pairs(
    nodes=nodes,
    num_questions_per_chunk=2,
    llm=Settings.llm,
)

# # save it as a json file
# question_context_pairs.save_json("question_context_pairs.json")

# load the dataset from the json file
# question_context_pairs = EmbeddingQAFinetuneDataset.from_json("question_context_pairs.json")
# retriever = vector_store.as_retriever(similarity_top_k=4)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=hybrid_retriever
)


def display_results_retriever(name, eval_results):
    
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)
    
    full_df = pd.DataFrame(metric_dicts)
    
    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()
    
    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
    )

    return metric_df



#create a retriever evaluator
async def data():
    eval_results = await retriever_evaluator.aevaluate_dataset(question_context_pairs)
    print(display_results_retriever("Retriever", eval_results))

import asyncio
asyncio.run(data())


# response evaluator
# get the list of queries from the question_context_pairs
queries = list(question_context_pairs.queries.values())

# create gpt-3.5-turbo model
gpt_3_5_turbo = OpenAI(model="gpt-3.5-turbo", temperature=0)
service_context_gpt_3_5_turbo = ServiceContext.from_defaults(llm=gpt_3_5_turbo)


# create a faithfulness evaluator using gpt4
faithfulness_gpt4 = FaithfulnessEvaluator(llm=Settings.llm)

# create a relevancy evaluator using gpt4
relevancy_gpt4 = RelevancyEvaluator(llm=Settings.llm)

# batch eval runner
batch_eval_runner = queries
# initialize the batch eval runner
runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=8,
)


# Compute evaluation
async def response_evaluator():
    eval_results = await runner.aevaluate_queries(
        query_engine, queries=batch_eval_runner
    )
    # lets get faithfulness and relevancy scores
    faithfulness_score =  sum(result.passing for result in eval_results['faithfulness']) / len(eval_results['faithfulness'])
    print(f"Faithfulness Score: {faithfulness_score}")

    relevancy_score =  sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])
    print(f"Relevancy Score: {relevancy_score}")
    

import asyncio
asyncio.run(response_evaluator())
