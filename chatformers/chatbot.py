from lazyme import color_print as cprint

try:
    from chroma_handler import create_client, recall, push_msg_to_vector_store
    from embedding_llm import get_embeddings
    from llm import run_llm
except:
    from chatformers.embedding_llm import get_embeddings
    from chatformers.chroma_handler import create_client, recall, push_msg_to_vector_store
    from chatformers.llm import run_llm


def chat(query, system_message,
         llm_provider_settings, chroma_settings, embedding_model_settings, memory_settings,
         memory=True,
         summarize_memory=False,
         collection_name=None,
         unique_session_id=None,
         unique_message_id=None,
         buffer_window_chats=None):
    """
    This is the entry point for the LLM Chatbot
    :param query: (str) current query that you want to ask LLM
    :param system_message: (str) the system message
    :param llm_provider_settings: (dict) the settings
    :param chroma_settings: (dict) the settings
    :param embedding_model_settings: (dict) the settings
    :param memory_settings: (dict) the settings
    :param memory: (bool) the memory settings if true then vector memory will be automatically managed using chroma db
    :param summarize_memory: (bool) flag to use summarize memory, improve quality but increase llm call
    :param collection_name: (str) chroma db collection name
    :param unique_session_id: (str) unique session id for the session using in chroma db
    :param unique_message_id: (str) unique message id using in chroma db
    :param buffer_window_chats: (list) list of chat messages in openai format excluding system message if None then no past conversation will be shown to model
    :return: Returns the chats response from Ollama or OpenAI
    """
    try:
        chroma_client = create_client(chroma_host=chroma_settings['host'],
                                      chroma_port=chroma_settings['port'],
                                      settings=chroma_settings['settings'])
        if memory:
            # If memory is enabled
            if collection_name is None or unique_session_id is None or unique_message_id is None:
                raise ValueError('Collection name, unique session id, and unique message id are required')

            # creating embeddings for query
            query_embedding = get_embeddings(embedding_model_settings=embedding_model_settings, text=query)

            # fetching memory / history of chats from vector database
            memories = recall(query, collection_name, unique_session_id, chroma_client,
                              query_embedding, try_queries=memory_settings['try_queries'],
                              results_per_query=memory_settings['results_per_query'],
                              llm_provider_settings=llm_provider_settings,
                              embedding_model_settings=embedding_model_settings)
            if summarize_memory:
                messages = [
                    {'role': 'system',
                     'content': f"\nSummarize given chats below- \n{memories}\n"},
                ]
                chat_response = run_llm(llm_provider_settings=llm_provider_settings, messages=messages)
                memories = chat_response

            # formatting the message system message, buffer_window_chats, user query and memories fetched in openai format
            if buffer_window_chats is None:
                buffer_window_chats = []
            messages = [
                           {'role': 'system',
                            'content': system_message + f"\nHere is the memory of old conversations-\n{memories}\n"},
                       ] + buffer_window_chats + [{'role': 'user', 'content': query}]

            # just printing the prompt
            for i in messages:
                cprint(f"{i['role']}: {i['content']}", color='cyan')

            chat_response = run_llm(llm_provider_settings=llm_provider_settings, messages=messages)
            # saving conversation just happened in vector store
            data = f"user: {query}\nassistant: {chat_response}"

            # creating embedding of conversation
            data_embedding = get_embeddings(embedding_model_settings=embedding_model_settings, text=data)

            # pushing conversation into vector database
            push_msg_to_vector_store(collection_name=collection_name, unique_session_id=unique_session_id,
                                     unique_message_id=unique_message_id, chroma_client=chroma_client,
                                     embeddings=data_embedding,
                                     data=data)
            return chat_response
        else:
            # if memory is set to False

            # converting system message and query to openai format
            if buffer_window_chats is None:
                buffer_window_chats = []

            messages = [
                           {'role': 'system', 'content': system_message},
                       ] + buffer_window_chats + [{'role': 'user', 'content': query}]

            chat_response = run_llm(llm_provider_settings=llm_provider_settings, messages=messages)
            return chat_response
    except Exception as e:
        cprint(f"An error occurred: {str(e)}", color='red')
        return str(e)


if __name__ == '__main__':
    # llm_provider_settings = {
    #     "provider": 'ollama',
    #     "base_url": 'http://localhost:11434',
    #     "model": "openhermes",
    #     "options": {},
    #     "api_key": None
    # }
    # embedding_model_settings = {
    #     "provider": 'ollama',
    #     "base_url": 'http://localhost:11434',
    #     "model": "nomic-embed-text",
    #     "api_key": None
    # }
    # llm_provider_settings = {
    #     "provider": 'openai',
    #     "base_url": "https://api.openai.com/v1",
    #     "model": "gpt-4o-mini",
    #     "options": {},
    #     "api_key": ""
    # }
    # embedding_model_settings = {
    #     "provider": 'openai',
    #     "base_url": "https://api.openai.com/v1",
    #     "model": "text-embedding-ada-002",
    #     "api_key": ""
    # }

    llm_provider_settings = {
        "provider": 'groq',
        "base_url": 'https://api.groq.com/openai/v1',
        "model": "gemma2-9b-it",
        "api_key": "",
    }

    embedding_model_settings = {
        "provider": 'jina',
        "base_url": "https://api.jina.ai/v1/embeddings",
        "model": "jina-embeddings-v2-base-en",
        "api_key": ""
    }

    chroma_settings = {
        "host": None,
        "port": None,
        "settings": None
    }

    memory_settings = {
        "try_queries": True,
        "results_per_query": 3,
    }
    collection_name = "conversation"
    unique_session_id = "012"
    unique_message_id = "A01"
    system_message = "You are a helpful assistant."
    buffer_window_chats = [
        {'role': 'user', 'content': 'what is 7*5?'},
        {'role': 'assistant', 'content': '35'},
        {'role': 'user', 'content': 'now add 4 on that.'},
    ]
    query = "Now add, 100 on that."
    response = chat(query=query, system_message=system_message,
                    llm_provider_settings=llm_provider_settings,
                    chroma_settings=chroma_settings,
                    embedding_model_settings=embedding_model_settings,
                    memory_settings=memory_settings,
                    memory=True,
                    collection_name=collection_name,
                    unique_session_id=unique_session_id,
                    unique_message_id=unique_message_id,
                    buffer_window_chats=buffer_window_chats)
    print("Assistant: ", response)
