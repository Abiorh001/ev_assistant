import os
import textwrap
from dotenv import load_dotenv
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import VectorStoreIndex, StorageContext, download_loader, get_response_synthesizer
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor


import re

# Load environment variables
load_dotenv()

# Fetch and set API keys
openai_api_key = os.getenv("OPENAI_API_KEY")


def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)


def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)


def initialize_github_client():
    github_token = os.getenv("GITHUB_TOKEN")
    return GithubClient(github_token)


def main():
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OpenAI API key not found in environment variables")

    # Check for GitHub Token
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise EnvironmentError("GitHub token not found in environment variables")

    # Check for Activeloop Token
    active_loop_token = os.getenv("ACTIVELOOP_TOKEN")
    if not active_loop_token:
        raise EnvironmentError("Activeloop token not found in environment variables")

    github_client = initialize_github_client()
    download_loader("GithubRepositoryReader")

    github_url = input("Please enter the GitHub repository URL: ")
    owner, repo = parse_github_url(github_url)

    while True:
        owner, repo = parse_github_url(github_url)
        if validate_owner_repo(owner, repo):
            loader = GithubRepositoryReader(
                github_client,
                owner=owner,
                repo=repo,
                filter_file_extensions=(
                    [".py", ".js", ".ts", ".md"],
                    GithubRepositoryReader.FilterType.INCLUDE,
                ),
                verbose=False,
                concurrent_requests=5,
            )
            print(f"Loading {repo} repository by {owner}")
            docs = loader.load_data(branch="main")
            print("Documents uploaded:")
            for doc in docs:
                print(doc.metadata)
            break  # Exit the loop once the valid URL is processed
        else:
            print("Invalid GitHub URL. Please try again.")
            github_url = input("Please enter the GitHub repository URL: ")

    print("Uploading to vector store...")

    # ====== Create in memory vector store and upload data ======

    index = VectorStoreIndex.from_documents(
        docs,
        show_progress=True,)
    # create a query engine
    # query_engine = index.as_query_engine()
    # creating a retriever and customizing the query engine
    retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
    response_synthesizer = get_response_synthesizer(streaming=True, response_mode='compact')
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessor=[
            SimilarityPostprocessor(similarity_cutoff=0.7),
        ]
    )

    # Include a simple question to test.
    intro_question = "What is the repository about?"
    print(f"Test question: {intro_question}")
    print("=" * 50)
    answer = query_engine.query(intro_question)

    print(f"Answer: {textwrap.fill(str(answer), 100)} \n")
    while True:
        user_question = input("Please enter your question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            print("Exiting, thanks for chatting!")
            break

        print(f"\nYour question: {user_question}")
        print("=" * 50)

        answer = query_engine.query(user_question)
        answer.print_response_stream()
        #print(f"Answer: {textwrap.fill(str(answer), 100)} \n")


if __name__ == "__main__":
    main()
