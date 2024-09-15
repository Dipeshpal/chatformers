from ollama import Client as OllamaClient
from openai import OpenAI
from constant import SUPPORTED_LLM


def run_llm(llm_provider_settings, messages):
    try:
        chat_model_name = llm_provider_settings['model']
        base_url = llm_provider_settings['base_url']
        provider = llm_provider_settings['provider']
        api_key = llm_provider_settings['api_key']
        if provider not in SUPPORTED_LLM:
            raise ValueError(f"Unsupported LLM provider: {provider}. We support only {SUPPORTED_LLM}")
        if provider == 'ollama':
            ollama_client = OllamaClient(host=base_url)
            response = ollama_client.chat(model=chat_model_name, messages=messages,
                                          options={'num_predict': 90})
            response = response['message']['content']
            return response
        if provider == 'openai' or provider == 'groq':
            client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            response = client.chat.completions.create(
                messages=messages,
                model=chat_model_name,
            )
            response = response.choices[0].message.content
            return response
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. We support only ollama and openai")
    except Exception as e:
        print(f"An error occurred while running LLM: {str(e)}")
        raise e


if __name__ == "__main__":
    # llm_provider_settings = {
    #     "provider": 'groq',
    #     "base_url": 'https://api.groq.com/openai/v1',
    #     "chat_model_name": "gemma2-9b-it",
    #     "api_key": "",
    # }
    # llm_provider_settings = {
    #     "provider": 'ollama',
    #     "base_url": '127.0.0.1:11434',
    #     "chat_model_name": "openhermes",
    #     "api_key": "",
    # }
    # messages = [
    #     {"role": "user", "content": "What is the weather like today?"},
    # ]
    # print(run_llm(llm_provider_settings, messages))
    pass
