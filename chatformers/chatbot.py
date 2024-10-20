import pprint
import re
import time
from openai import OpenAI
from mem0 import Memory
import os
from lazyme import color_print as cprint
from enum import Enum
import logging

logging.basicConfig(level=logging.DEBUG)

try:
    from constants import *
except:
    from chatformers.constants import *


# Enum for message types
class MessageType(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


# Retry function for any operation that may fail
def retry_operation(operation, max_retries=3, retry_delay=2, **kwargs):
    """
    Generic retry function to attempt an operation multiple times in case of failure.

    :param operation: The function to attempt.
    :param max_retries: Maximum number of retries (default is 3).
    :param retry_delay: Delay between retries in seconds (default is 2).
    :param kwargs: Arguments to pass to the operation.
    :return: The result of the operation or raises an exception after max retries.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            return operation(**kwargs)
        except Exception as e:
            attempt += 1
            if attempt < max_retries:
                logging.error(f"Attempt {attempt} failed with error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logging.info(f"All {max_retries} attempts failed.")
                raise e


# Function to clean text by fixing spacing around punctuation, parentheses, and quotes
def clean_text(text):
    """
    Cleans the input text by ensuring proper spacing after punctuation and before contractions.

    :param text: The input string to be cleaned.
    :return: The cleaned text.
    """
    patterns = [
        (r'([.!?])([A-Za-z])', r'\1 \2'),  # Ensure space after punctuation (.,!?) if followed by a letter
        (r'\s+([.!?])', r'\1'),  # Remove extra spaces before punctuation
        (r'(\w)\s*,\s*', r'\1, '),  # Ensure space after commas
        (r'\(\s+', '('),  # Remove spaces after opening parenthesis
        (r'\s+\)', ')'),  # Remove spaces before closing parenthesis
        (r'\s+"', '"'),  # Remove unnecessary spaces around quotes
        (r'"\s+', '"'),  # Remove unnecessary spaces around quotes
        (r"\b(\w+)\s+'\s+(\w+)\b", r"\1'\2"),
        # Remove space before contractions with single quotes (e.g., "We 've" -> "We've")
    ]

    # Loop through patterns and apply each regex substitution on the text
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)

    return text


class Chatbot:
    def __init__(self, config: dict = None, llm_client=None, metadata: dict = None, system_prompt=None,
                 chat_model_name=None, memory_model_name=None, max_tokens=DEFAULT_MAX_TOKENS, limit=DEFAULT_LIMIT,
                 debug=False):
        """
        Initialize the Chatbot class with necessary configurations.

        :param config: Configuration dictionary (default uses predefined settings if None).
        :param llm_client: OpenAI LLM compatible client for interacting with the AI model.
        :param metadata: Metadata dictionary to store additional information about the chat.
        :param system_prompt: Initial system prompt for the chat (e.g., instructions for the chatbot).
        :param chat_model_name: The LLM model name for handling chat interactions.
        :param memory_model_name: The LLM model name for managing memory-based interactions.
        :param max_tokens: Max token length for generating chat responses.
        :param limit: Limit on the number of memories to use during conversation.
        :param debug: Enable debugging mode for extra logging.
        """
        self.custom_prompt = DEFAULT_CUSTOM_MEMORY_PROMPT
        self.memory_model_name = DEFAULT_GROQ_MODEL_MEMORY if memory_model_name is None else memory_model_name
        self.chat_model_name = DEFAULT_GROQ_MODEL_CHAT if chat_model_name is None else chat_model_name

        # Default configuration setup if not provided
        if config is None:
            self.config = {
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
                        "model": self.memory_model_name,
                        "temperature": 0.1,
                        "max_tokens": 1000,
                    }
                },
                "custom_prompt": self.custom_prompt
            }
            cprint("Using default Configuration for memory.", color='cyan')
        else:
            self.config = config

        # Initialize memory and llm_client
        self.memory = Memory.from_config(self.config)
        if llm_client is None:
            raise ValueError("LLM Client `llm_client` is not available. Please set it first.")
        self.client = llm_client
        self.metadata = metadata or {}
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.limit = limit
        self.debug = debug

        # Display configuration in debug mode
        self._print_debug(text="USING BELOW GIVEN CONFIGS-", color="cyan", message_type=MessageType.INFO)
        pprint.pprint(self.config)
        self._print_debug(text="END OF CONFIGS", color="cyan", message_type=MessageType.INFO)

    def _print_debug(self, text, message_type: MessageType, color="yellow"):
        """Helper function to print debug information."""
        if self.debug:
            cprint(f"{message_type.value}: {text}", color=color)

    def chat(self, query, message_history, user_id=None, print_stream=False):
        """
        Handle a chat query and process it through the chatbot.

        :param query: The user query.
        :param message_history: List of previous messages.
        :param user_id: User ID associated with the memory.
        :param print_stream: If True, print the response from the chatbot in real-time.
        :return: Final chatbot response.
        """
        if not user_id:
            raise ValueError("You must provide a user_id associated with the memory.")
        memories = self.related_memory(query, user_id)
        self.system_prompt += f"{DEFAULT_MEMORY_PROMPT}\n{memories}"
        self._print_debug(text=f"SYSTEM PROMPT-\n{self.system_prompt}", color="yellow", message_type=MessageType.INFO)

        # Make the chat request to the AI model
        stream = self.client.chat.completions.create(
            max_tokens=self.max_tokens,
            model=self.chat_model_name,
            stream=True,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *message_history,
                {"role": "user", "content": query}
            ]
        )

        # Process and clean the response in real-time
        if print_stream:
            cprint("Assistant: ", color="green", end="")
        final_response = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                if final_response and re.match(r'[.!?,]', content.strip()):
                    final_response = final_response.rstrip() + content.strip()
                else:
                    final_response += content.strip() + " "
                if print_stream:
                    cprint(content, color="green", end="")

        final_response = clean_text(final_response.strip())
        print("\n")
        # Add the conversation to memory
        memory_data = [{"role": "user", "content": query}, {"role": "assistant", "content": final_response}]
        self.add_memories(messages=memory_data, user_id=user_id)

        return final_response

    def related_memory(self, query: str, user_id: str, max_retries: int = 3, retry_delay: int = 1):
        """
        Retrieve related memories for a given query and user.

        :param query: Query to search memories.
        :param user_id: Associated user ID.
        :param max_retries: Number of retries if the operation fails.
        :param retry_delay: Delay between retries.
        :return: Retrieved memories.
        """
        return retry_operation(self._search_memories, max_retries, retry_delay, query=query, user_id=user_id)

    def _search_memories(self, query, user_id):
        """Helper function to search memories."""
        related_memories = self.memory.search(query=query, user_id=user_id, limit=self.limit)
        return "\n".join(m['memory'] for m in related_memories)

    def get_memories(self, user_id: str = None, max_retries: int = 3, retry_delay: int = 2):
        """Retrieve all memories for a specific user."""
        return retry_operation(self.memory.get_all, max_retries, retry_delay, user_id=user_id)

    def add_memories(self, messages: list[dict], user_id: str, max_retries: int = 3, retry_delay: int = 2) -> None:
        """Add a list of messages to the user's memory."""
        for message in messages:
            retry_operation(self._add_memory, max_retries, retry_delay, message=message, user_id=user_id)

    def _add_memory(self, message, user_id) -> None:
        """Helper function to add a single message to memory."""
        updated_metadata = self.metadata.copy()
        updated_metadata.update({"role": message['role']})
        self.memory.add(f"{message['role']} input: {message['content']}", user_id=user_id, metadata=updated_metadata)


if __name__ == "__main__":
    from chatformers.chatbot import Chatbot
    import os
    from openai import OpenAI

    system_prompt = None  # use the default
    metadata = None  # use the default metadata
    user_id = "Sam-Julia"
    chat_model_name = "llama-3.1-8b-instant"
    memory_model_name = "llama-3.1-8b-instant"
    max_tokens = 150  # len of tokens to generate from LLM
    limit = 4  # maximum number of memory to added during LLM chat
    debug = True  # enable to print debug messages

    os.environ["GROQ_API_KEY"] = ""
    llm_client = OpenAI(base_url="https://api.groq.com/openai/v1",
                        api_key="",
                        )  # Any OpenAI Compatible LLM Client, using groq here
    config = {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": user_id,
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
    query = "Do you remember my name?"
    response = chatbot.chat(query=query, message_history=message_history, user_id=user_id, print_stream=True)
    print("Assistant: ", response)

    # Example to check memories in bot based on user_id
    # memories = chatbot.get_memories(user_id=user_id)
    # for m in memories:
    #     print(m)
    # print("================================================================")
    # related_memories = chatbot.related_memory(user_id=user_id,
    #                                           query="yes i am sam? what us your name")
    # print(related_memories)
