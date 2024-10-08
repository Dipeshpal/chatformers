import ast
import chromadb
from tqdm import tqdm

try:
    from llm import run_llm
    from embedding_llm import get_embeddings
except:
    from .llm import run_llm
    from .embedding_llm import get_embeddings


def create_client(chroma_host: str = None, chroma_port: int = None, settings=None):
    """
    This function creates a chroma client instance
    :param chroma_host: (str) The host of the chroma server optional, if None then it will create persistent client
    :param chroma_port: (int) The port of the chroma server optional, if None then it will create persistent client
    :param settings: (dict) The settings for the chroma client optional, if None then it will create persistent client
    :return: (object) The chroma client instance
    """
    if chroma_host is None or chroma_port is None:
        chroma_client = chromadb.PersistentClient()
    else:
        chroma_client = chromadb.HttpClient(host=chroma_host,
                                            port=chroma_port,
                                            settings=settings,
                                            )
    return chroma_client


def push_msg_to_vector_store(collection_name: str, unique_session_id, unique_message_id, chroma_client, embeddings,
                             data):
    """
    This function is used to push message embeddings to the vector store
    :param collection_name: (str) The name of the collection
    :param unique_session_id: (str) The unique session
    :param unique_message_id: (str) The unique message
    :param chroma_client: (object) The chroma client instance
    :param embeddings: (list) The message embeddings
    :param data: (str) The message data you want to push to the vector store
    :return: (bool) True if successful push to the vector store and false otherwise
    """
    try:
        vector_db_name = f'{collection_name}-{unique_session_id}'
        vector_db = chroma_client.get_or_create_collection(name=vector_db_name)
        vector_db.add(
            ids=[unique_message_id],
            embeddings=[embeddings],
            documents=[data],
            metadatas=[{"collection_name": collection_name,
                        "unique_session_id": unique_session_id,
                        "unique_message_id": unique_message_id,
                        }]
        )
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def create_queries(prompt, llm_provider_settings):
    """
    This will create refine queries to search in vector database, this can be user for retrieving from vector database
    :param prompt: (str) Original query / msg / prompt from user end
    :param llm_provider_settings: (dict) llm provider settings
    :return: (list) List of refine queries
    """
    query_msg = (
        'You are a first principle reasoning search query AI agent.'
        'Your list of search queries will be ran on an embedding database of all your conversations '
        'you have ever had conversation with the user. With first principles create a Python list of length 2 with queries to '
        'search the embeddings database for any data that would be necessary to have access to in '
        'order to correctly respond to the prompt. Your response must be a Python list of 2 queries with no syntax errors.'
        'Do not explain anything and do not ever generate anything but a perfect syntax Python list of 2 queries only.'
    )

    query_convo = [
        {'role': 'system', 'content': query_msg},
        {'role': 'user',
         'content': 'write an email to my car insurance company and create and create a persuasive request for them to lower my monthly rate.'},
        {'role': 'assistant',
         'content': '["what is the user name?", "what is the users current auto insurance provider", "what is the monthly rate the user currently pays for auto insurance?"]'},
        {'role': 'user', 'content': 'what did you told to Smith?'},
        {'role': 'assistant',
         'content': '["Who is smith?", "How Smith is related to user?", "Is Smith is nickname of user?", "What is assistant nickname?"]'},
        {'role': 'user', 'content': prompt}
    ]
    response = run_llm(llm_provider_settings=llm_provider_settings, messages=query_convo)

    try:
        queries_ = ast.literal_eval(response) + [prompt]
        if len(queries_) > 3:
            queries_ = queries_[-3:]
        print(f'\nVector Database Queries sliced: {queries_}\n')
        return queries_
    except Exception as e:
        print(f"Warning: {e}")
        print(f'\nVector Database Queries: {[prompt]}\n')
        return [prompt]


def retrieve_embeddings(collection_name: str, unique_session_id, chroma_client,
                        query_embedding,
                        results_per_query=2):
    embeddings = list()
    vector_db_name = f'{collection_name}-{unique_session_id}'
    vector_db = chroma_client.get_or_create_collection(name=vector_db_name)
    results = vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
    best_embeddings = results['documents'][0]
    for best in best_embeddings:
        embeddings.append(best)
    return embeddings


def retrieve_embeddings_try_queries(queries, collection_name: str, unique_session_id, chroma_client,
                                    results_per_query=2, embedding_model_settings=None):
    """
    This function is used to retrieve embeddings (memories as list of strings) from vector databas
    :param queries: (list) list of queries
    :param collection_name: (str) name of the collection of vector database
    :param unique_session_id: (str) unique session id for vector database
    :param chroma_client: (obj) chroma client object
    :param results_per_query: (int) number of similar docs to fetch
    :param embedding_model_settings: (dict) dictionary of embeddings models settings
    :return: (list) list of memories as list of strings
    """
    embeddings = list()
    vector_db_name = f'{collection_name}-{unique_session_id}'

    for query in tqdm(queries, desc='Processing queries to vector database'):
        query_embedding = get_embeddings(embedding_model_settings, text=query)
        vector_db = chroma_client.get_or_create_collection(name=vector_db_name)
        results = vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
        best_embeddings = results['documents'][0]
        for best in best_embeddings:
            embeddings.append(best)
    print(f"{len(embeddings)} past conversation fetched.'")
    return embeddings


def recall(query, collection_name, unique_session_id, chroma_client,
           query_embedding, try_queries=False, results_per_query=2, llm_provider_settings=None,
           embedding_model_settings=None):
    """
    This function is used to recall memories from vector database
    :param query: (str) original query
    :param collection_name: (str) name of the collection of vector database
    :param unique_session_id: (str) unique session id for vector database
    :param chroma_client: (obj) chroma client object
    :param query_embedding: (list) embedding of the query
    :param try_queries: (bool) flag to use refined queries
    :param results_per_query: (int) number of similar docs to fetch
    :param llm_provider_settings: (dict) dictionary of llm settings
    :param embedding_model_settings: (dict) dictionary of embeddings models settings
    :return: (dict) dictionary with memories
    """
    if try_queries:
        queries = create_queries(prompt=query, llm_provider_settings=llm_provider_settings)
        embeddings = retrieve_embeddings_try_queries(queries, collection_name, unique_session_id, chroma_client,
                                                     results_per_query=results_per_query,
                                                     embedding_model_settings=embedding_model_settings)
    else:
        embeddings = retrieve_embeddings(collection_name, unique_session_id, chroma_client,
                                         query_embedding,
                                         results_per_query=results_per_query)
    memories = {'memories': f'{embeddings}'}
    return memories
