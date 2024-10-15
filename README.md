# Chatformers

⚡ Chatformers is a Python package designed to simplify the development of chatbot applications that use Large Language Models (LLMs). It offers automatic chat history management using a local vector database (ChromaDB, Qdrant or Pgvector), ensuring efficient context retrieval for ongoing conversations.

![Static Badge](https://img.shields.io/badge/license-MIT?style=for-the-badge&label=MIT&link=https%3A%2F%2Fopensource.org%2Flicense%2FMIT)
[![Release Notes](https://img.shields.io/github/release/Dipeshpal/chatformers?style=flat-square)](https://github.com/Dipeshpal/chatformers/releases)

# Install

```
pip install chatformers
```

# Documentation-

https://coda.io/@chatformers/chatformers

## Why Choose chatformers?
1. Effortless History Management: No need to manage extensive chat history manually; the package automatically handles it.
2. Simple Integration: Build a chatbot with just a few lines of code.
3. Full Customization: Maintain complete control over your data and conversations.
4. Framework Compatibility: Easily integrate with any existing framework or codebase.


## Key Features
1. Easy Chatbot Creation: Set up a chatbot with minimal code.
2. Automated History Management: Automatically stores and fetches chat history for context-aware conversations.

## How It Works
1. Project Setup: Create a basic project structure.
2. Automatic Storage: Chatformers stores your conversations (user inputs and AI outputs) in VectorDB.
3. Contextual Conversations: The chatbot fetches relevant chat history whenever you engage with the LLM.


## Prerequisites-

1. Python: Ensure Python is installed on your system.
2. GenAI Knowledge: Familiarity with Generative AI models.

## Example Usage-

Read Documentation for advanced usage and understanding: https://coda.io/@chatformers/chatformers

```
from chatformers.chatbot import Chatbot
import os
from openai import OpenAI

os.environ["GROQ_API_KEY"] = "<API_KEY>"
GROQ_API_KEY = "<API_KEY>"
groq_base_url = "https://api.groq.com/openai/v1"

# Unique ID for conversation between Sam (User) and Julia (Chatbot)
user_id = "Sam-Julia"

# Name of the model you want to use
model_name = "llama-3.1-8b-instant"

# Initialize OpenAI client with API key and base URL, we are using LLM from GROQ here, this is required for having conversation with LLM
client = OpenAI(base_url=groq_base_url,
                api_key=GROQ_API_KEY,
                )

# You can provide character to your chatbot, the type should be dictionary with key value pairs of your choice we will integrate in system prompt or you can leave it empty dictionary
character_data = {"name": "Julia",
                  "description": "You are on online chatting website, chatting with strangers."}

# Configuration: for configuration you can refer https://docs.mem0.ai/overview, hence chatformers use mem0 for memory and llm management
# Example: https://docs.mem0.ai/examples/mem0-with-ollama
# These configuration will be used for embedded the chats, handling memory creation automatically
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": "db",
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest"
        }
    },
    "llm": {
        "provider": "groq",
        "config": {
            "model": model_name,
            "temperature": 0.1,
            "max_tokens": 4000,
        }
    },
    # "llm": {
    #     "provider": "ollama",
    #     "config": {
    #         "model": model_name,
    #         "temperature": 0.1,
    #         "max_tokens": 4000,
    #     }
    # },
}

# Initialize Chatbot with LLM client, model name, character data, and configuration
chatbot = Chatbot(llm_client=client, model_name=model_name, character_data=character_data, config=config)

# Optional, if you want to add any memory into vector database at any point, uncomment this line
# memory_messages = [
#     {"role": "user", "content": "My name is Sam, what about you?"},
#     {"role": "assistant", "content": "Hello Sam! I'm Julia."}
# ]
# chatbot.add_memories(memory_messages, user_id=user_id)

# query is your current question that you want LLM to answer
query = "what is my name"

# message_history is a list of messages in openai format, this can be your conversation buffer window memory, you can manage it yourself
message_history = [{"role": "user", "content": "where r u from?"},
                   {"role": "assistant", "content": "I am from CA, USA"}]
response = chatbot.chat(query=query, message_history=message_history, user_id=user_id,
                        print_stream=True)

# Final response from LLM based on message_history, and memory you have added if any and whatever chats happened with user_id
print("Assistant: ", response)

# Optional, Uncomment this line to get all memories of a user
# memories = chatbot.get_memories(user_id=user_id)
# for m in memories:
#     print(m)
# print("================================================================")
# related_memories = chatbot.related_memory(user_id=user_id,
#                                           query="yes i am sam? what us your name")
# print(related_memories)
```



## FAQs-

1. Can I customize LLM endpoints / Groq or other models?
    - Yes, any OpenAI-compatible endpoints and models can be used.

2. Can I use custom hosted chromadb
    - Yes, you can specify custom endpoints for Chroma DB. If not provided, a Chroma directory will be created in your project's root folder.

3. I don't want to manage history. Just wanted to chat.
    - Yes, set `memory=False` to disable history management and chat directly.
 
4. Need help or have suggestions?
    - Raise an issue or contact me at dipesh.paul@systango.com


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dipeshpal/chatformers&type=Date)](https://star-history.com/#Dipeshpal/chatformers&Date)
