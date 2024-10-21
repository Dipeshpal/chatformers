DEFAULT_CUSTOM_MEMORY_PROMPT = """Your task is to extract facts from the conversations between the user and the assistant. Focus on personal details, preferences, and contextual information shared by either the user or the assistant. Return the information in JSON format with the key "facts", which should contain a list of statements based on the conversation.

Few-shot Examples:
Example 1:
User Input: Hi, I am Avinash, what about you?
Output: {"facts": ["The user's name is Avinash"]}

Example 2:
Assistant Input: Hello, I am Sophia. Where are you from?
Output: {"facts": ["The assistant's name is Sophia"]}

Example 3:
User Input: I am from Mumbai. Where are you from?
Output: {"facts": ["The user is from Mumbai, India"]}

Example 4:
Assistant Input: I am from CA, USA. I like to visit Mumbai.
Output: {"facts": ["The assistant is from California, USA", "The assistant wants to visit Mumbai, India"]}

Output Format:
- Return all extracted facts as a JSON object with a single key "facts", containing a list of strings.
- Each string should be a distinct factual statement derived from the conversation.
- Ensure that factual information about both the user and the assistant is captured.

Final Output Example:

{
  "facts": [
    "The user's name is Avinash",
    "The user is from Mumbai, India",
    "The assistant's name is Sophia",
    "The assistant is from California, USA",
    "The assistant wants to visit Mumbai, India"
  ]
}


Important Notes:

Focus on key details shared by the user or assistant, such as names, locations, preferences, and relevant personal information, general conversation or anything.
Maintain consistency in how facts are phrased, ensuring clear and accurate statements.
Use proper sentence structure and avoid any abbreviations in the output.
"""

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

DEFAULT_GROQ_MODEL_CHAT = "llama-3.1-8b-instant"
DEFAULT_GROQ_MODEL_MEMORY = "llama-3.1-8b-instant"
DEFAULT_MAX_TOKENS = 150
DEFAULT_MEMORY_PROMPT = "You have access of following memories from old conversation you had earlier. You can refer these if required-"
DEFAULT_LIMIT = 4
