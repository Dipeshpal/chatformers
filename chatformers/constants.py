DEFAULT_CUSTOM_MEMORY_PROMPT = """Please only extract entities containing conversation between user and assistant. 
Here are some few shot examples:

user Input: Hi, I am Avinash, what about you.
Output: {{"facts" : ["user's name is avinash"]}}

assistant Input: Hello, I am Sophia, where r u from?
Output: {{"facts" : ["assistant's name is Sophia'"]}}

user Input: I am from Mumbai, where r u from?
Output: {{"facts" : ["user is from Mumbai, India"]}}

assistant Input: I am from CA, USA. I like to vist Mumbai.
Output: {{"facts" : ["assistant is from CA, USA", "Assistant wants to vist Mumbai, India"]}}


Output Format: json
Return the facts and user / assistant information in a json format as shown above.
Information in a json format as shown above.
"""

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

DEFAULT_GROQ_MODEL_CHAT = "llama-3.1-8b-instant"
DEFAULT_GROQ_MODEL_MEMORY = "llama-3.1-8b-instant"
DEFAULT_MAX_TOKENS = 150
DEFAULT_MEMORY_PROMPT = "You have access of following memories from old conversation you had earlier. You can refer these if required-"
DEFAULT_LIMIT = 4
