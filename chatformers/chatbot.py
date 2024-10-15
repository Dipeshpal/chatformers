import re
from importlib.metadata import metadata
from openai import OpenAI
from mem0 import Memory
import os


def clean_text(text):
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.!?,])', r'\1', text)
    # Remove spaces inside parentheses
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    # Remove unnecessary spaces around quotes
    text = re.sub(r'\s+"', '"', text)
    text = re.sub(r'"\s+', '"', text)
    # Fix contractions (don 't -> don't)
    text = re.sub(r"\b(\w+)\s+'\s+(\w+)\b", r"\1'\2", text)
    return text


class Chatbot:
    def __init__(self, character_data, config: dict = None, llm_client=None, metadata: dict = None,
                 system_prompt: str = None,
                 model_name=None, limit=2):
        """
        Chatformers base class for Chat
        :character_data: character data / persona of assistant
        :param config: Configuration dictionary
        :param llm_client: OpenAI LLM compatible llm client or similar
        :param metadata: Metadata dictionary
        :param system_prompt: System prompt to use in the chat
        :param model_name: The name of the LLM model to use. Default is 'llama-3.1-8b-instant'.
        :param limit: Maximum number of memory to use in LLM for communication
        """
        self.character_data = character_data
        self.custom_prompt = """
               Please only extract entities containing conversation between user and assistant. 
               Here are some few shot examples:

               user Input: Hi, I am Avinash, what about you.
               Output: {{"facts" : ["user's name is avinash"]}}

               assistant Input: Hello, I am Sophia, where r u from?
               Output: {{"facts" : ["assistant's name is Sophia'"]}}

               user Input: I am from Mumbai, where r u from?
               Output: {{"facts" : ["user is from Mumbai, India"]}}

               assistant Input: I am from CA, USA. I like to vist Mumbai.
               Output: {{"facts" : ["assistant is from CA, USA", "Assistant wants to vist Mumbai, India"]}}

               Return the facts and user / assistant information in a json format as shown above.
               
               Output Format: json
               """
        if config is None:
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
                        "max_tokens": 1000,
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
                "custom_prompt": self.custom_prompt
            }
            print("Using default Configuration.")
        self.memory = Memory.from_config(config)
        if llm_client is None:
            raise ValueError("LLM Client `llm_client` is not available. Please set it first.")
        self.client = llm_client
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
        if system_prompt is None:
            self.system_prompt = "You are a helpful assistant. "
            print("Using default system_prompt")
        else:
            self.system_prompt = system_prompt
            print("Using provided system_prompt")
        if model_name is None:
            self.model_name = "llama-3.1-8b-instant"
            print("Using default LLM model")
        else:
            self.model_name = model_name
            print("Using provided LLM model")
        self.limit = limit

    def chat(self, query, message_history, user_id=None, print_stream=False):
        """
        Handle a chat query and store the relevant information in memory.
        :param query: The customer query to handle.
        :param message_history: Previous messages from the chat in openai format.
                        Excluding system message and current message (last message / query).
                        This can you your buffer window memory.
        :param user_id: Optional user ID to associate with the memory.
        :param print_stream: If True, print the AI's response in real-time.
        :return: The final response from the AI.
        """

        if user_id is None:
            raise ValueError("You must provide user_id associated with the memory")
        memories = self.related_memory(query, user_id)
        name = self.character_data.get('name', "Julia")
        description = self.character_data.get('description',
                                              "You are on online chatting website, chatting with strangers.")
        self.system_prompt += ("This is your Persona, remember your persona-\n"
                               f"Name: {name}\n"
                               f"Description: {description}\n")
        self.system_prompt += (f"You have access of following memories from old conversation you had earlier. "
                               f"You can refer these if required-\n{memories}")
        print("SYSTEM PROMPT: ", self.system_prompt)
        stream = self.client.chat.completions.create(
            max_tokens=100,
            model=self.model_name,
            stream=True,
            messages=[
                         {"role": "system", "content": self.system_prompt},
                     ] + message_history + [{"role": "user", "content": query}]
        )

        # Print the response from the AI in real-time
        final_response = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                if final_response and re.match(r'[.!?,]', content.strip()):
                    final_response = final_response.rstrip() + content.strip()
                else:
                    final_response += content.strip() + " "

                if print_stream:
                    print(content, end="")

        # Apply the cleaning function to fix spacing issues
        final_response = clean_text(final_response.strip())

        memory_data = [{"role": "user", "content": query}, {"role": "assistant", "content": final_response}]
        self.add_memories(messages=memory_data, user_id=user_id)
        print("\n")
        return final_response

    def related_memory(self, query, user_id):
        """
        Retrieve related memories from the memory for the given query and user_id.
        :param query: The query to search for related memories.
        :param user_id: The user ID to associate with the memories.
        :return: A list of related memories.
        """
        related_memories = self.memory.search(query=query, user_id=user_id, limit=self.limit)
        related_memories_response = ""
        for m in related_memories:
            related_memories_response += m['memory'] + "\n"
        return related_memories_response

    def get_memories(self, user_id=None):
        """
        Retrieve all memories associated with the given customer ID.

        :param user_id: Optional user ID to filter memories.
        :return: List of memories.
        """
        return self.memory.get_all(user_id=user_id)

    def add_memories(self, messages, user_id):
        for i in messages:
            self.memory.add(f"{i['role']} Input: {i['content']}", user_id=user_id,
                            metadata=self.metadata.update({"role": i['role']}))


if __name__ == "__main__":
    user_id = "Sam-Julia"
    # model_name = ""
    model_name = "llama-3.1-8b-instant"
    os.environ["GROQ_API_KEY"] = "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"
    client = OpenAI(base_url="https://api.groq.com/openai/v1",
                    api_key="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",
                    )
    character_data = {"name": "Julia",
                      "description": "You are on online chatting website, chatting with strangers."}
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
                "max_tokens": 1000,
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
    chatbot = Chatbot(llm_client=client, model_name=model_name, character_data=character_data, config=config)

    # memory_messages = [
    #     {"role": "user", "content": "My name is Sam, what about you?"},
    #     {"role": "assistant", "content": "Hello Sam! I'm Julia."}
    # ]
    # chatbot.add_memories(memory_messages, user_id=user_id)

    query = "then which food do you like?"
    message_history = [{"role": "user", "content": "where r u from?"},
                       {"role": "assistant", "content": "I am from CA, USA"}]
    response = chatbot.chat(query=query, message_history=message_history, user_id=user_id,
                            print_stream=True)
    print("Assistant: ", response)

    # memories = chatbot.get_memories(user_id=user_id)
    # for m in memories:
    #     print(m)
    # print("================================================================")
    # related_memories = chatbot.related_memory(user_id=user_id,
    #                                           query="yes i am sam? what us your name")
    # print(related_memories)
