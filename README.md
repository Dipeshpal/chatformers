# Chatformers

⚡ Chatformers is a Python package designed to simplify the development of chatbot applications that use Large Language Models (LLMs). It offers automatic chat history management using a local vector database (ChromaDB, Qdrant or Pgvector), ensuring efficient context retrieval for ongoing conversations.

![Static Badge](https://img.shields.io/badge/license-MIT?style=for-the-badge&label=MIT&link=https%3A%2F%2Fopensource.org%2Flicense%2FMIT)
[![Release Notes](https://img.shields.io/github/release/Dipeshpal/chatformers?style=flat-square)](https://github.com/Dipeshpal/chatformers/releases)

# Install

```
pip install chatformers
```

# Documentation-

https://chatformers.mintlify.app/introduction

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

Read Documentation for advanced usage and understanding: https://chatformers.mintlify.app/development

```
   from chatformers.chatbot import Chatbot
   import os
   from openai import OpenAI
   
   
   system_prompt = None  # use the default
   metadata = None  # use the default metadata
   user_id = "Sam-Julia"
   chat_model_name = "llama-3.1-70b-versatile"
   memory_model_name = "llama-3.1-70b-versatile"
   max_tokens = 150  # len of tokens to generate from LLM
   limit = 4  # maximum number of memory to added during LLM chat
   debug = True  # enable to print debug messages
   
   os.environ["GROQ_API_KEY"] = ""
   llm_client = OpenAI(base_url="https://api.groq.com/openai/v1",
                       api_key="",
                       )  # Any OpenAI Compatible LLM Client
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
               "model": memory_model_name,
               "temperature": 0.1,
               "max_tokens": 1000,
           }
       },
   }
   
   chatbot = Chatbot(config=config, llm_client=llm_client, metadata=None, system_prompt=system_prompt,
                     chat_model_name=chat_model_name, memory_model_name=memory_model_name,
                     max_tokens=max_tokens, limit=limit, debug=debug)
   
   # Example to add buffer memory
   memory_messages = [
       {"role": "user", "content": "My name is Sam, what about you?"},
       {"role": "assistant", "content": "Hello Sam! I'm Julia."},
       {"role": "user", "content": "What do you like to eat?"},
       {"role": "assistant", "content": "I like pizza"}
   ]
   chatbot.add_memories(memory_messages, user_id=user_id)
   
   # Buffer window memory, this will be acts as sliding window memory for LLM
   message_history = [{"role": "user", "content": "where r u from?"},
                      {"role": "assistant", "content": "I am from CA, USA"},
                      {"role": "user", "content": "ok"},
                      {"role": "assistant", "content": "hmm"},
                      {"role": "user", "content": "What are u doing on next Sunday?"},
                      {"role": "assistant", "content": "I am all available"}
                      ]
   # Example to chat with the bot, send latest / current query here
   query = "Could you remind me what do you like to eat?"
   response = chatbot.chat(query=query, message_history=message_history, user_id=user_id, print_stream=True)
   print("Assistant: ", response)
   
   # # Example to check memories in bot based on user_id
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

2. Can I use custom hosted chromadb, or any other vector db.
    - Yes, read documentation.

3. Need help or have suggestions?
    - Raise an issue or contact me at dipeshpal17@gmail.com


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dipeshpal/chatformers&type=Date)](https://star-history.com/Dipeshpal/chatformers&Date)
