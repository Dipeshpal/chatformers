from ollama import Client as OllamaClient
from openai import OpenAI
import requests

try:
    from constant import SUPPORTED_EMBEDDING_MODELS
except:
    from .constant import SUPPORTED_EMBEDDING_MODELS


def get_embeddings(embedding_model_settings, text):
    try:
        provider = embedding_model_settings['provider']
        api_key = embedding_model_settings['api_key']
        base_url = embedding_model_settings['base_url']
        embedding_model_name = embedding_model_settings['model']

        if provider not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(
                f"Unsupported embedding model provider: {provider}. Supported providers: {SUPPORTED_EMBEDDING_MODELS}"
            )

        if provider == 'ollama':
            try:
                llm_client = OllamaClient(host=base_url)
                response = llm_client.embeddings(model=embedding_model_name, prompt=text)
                query_embedding = response['embedding']
                return query_embedding
            except KeyError as e:
                print(f"Missing key in response from Ollama: {e}")
                raise e
            except Exception as e:
                print(f"Error fetching embeddings from Ollama: {e}")
                raise e

        if provider == 'openai':
            try:
                llm_client = OpenAI(api_key=api_key, base_url=base_url)
                response = llm_client.embeddings.create(
                    input=text,
                    model=embedding_model_name
                )
                query_embedding = response.data[0].embedding
                return query_embedding
            except KeyError as e:
                print(f"Missing key in response from OpenAI: {e}")
                raise e
            except Exception as e:
                print(f"Error fetching embeddings from OpenAI: {e}")
                raise e

        if provider == 'jina':
            try:
                url = base_url
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }
                data = {
                    "model": embedding_model_name,
                    "normalized": True,
                    "embedding_type": "float",
                    "input": [
                        {"text": text},
                    ]
                }
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()  # Check if request was successful
                embedding = response.json()['data'][0]['embedding']
                return embedding
            except requests.RequestException as e:
                print(f"HTTP request error from Jina: {e}")
                raise e
            except KeyError as e:
                print(f"Missing key in response from Jina: {e}")
                raise e
            except Exception as e:
                print(f"Error fetching embeddings from Jina: {e}")
                raise e

    except Exception as e:
        print(f"Error in get_embeddings function: {e}")
        raise e


if __name__ == "__main__":
    # embedding_model_settings = {
    #     "provider": 'ollama',
    #     "base_url": 'http://localhost:11434',
    #     "embedding_model_name": "nomic-embed-text",
    #     "api_key": None
    # }
    # embedding_model_settings = {
    #     "provider": 'jina',
    #     "base_url": 'https://api.jina.ai/v1/embeddings',
    #     "embedding_model_name": "jina-embeddings-v2-base-en",
    #     "api_key": ""
    # }
    # text = "What is the capital of France?"
    # embeddings = get_embeddings(embedding_model_settings, text)
    # pprint.pprint(embeddings)
    pass
