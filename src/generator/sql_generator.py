import os
import re
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)

MODEL_NAME = "models/gemini-2.5-flash"   

def clean_sql(response_text: str) -> str:
    # Remove markdown code blocks
    response_text = re.sub(r"```.*?```", lambda m: m.group(0).replace("```", ""), response_text, flags=re.DOTALL)

    # Remove leftover backticks or language hints
    response_text = response_text.replace("```sql", "")
    response_text = response_text.replace("```sqlite", "")
    response_text = response_text.replace("```", "")

    # Extract first SELECT statement
    match = re.search(r"(SELECT .*?;)", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: return cleaned text
    return response_text.strip()

def generate_sql(question: str, schema: str) -> str:
    prompt = f"""
You are a SQL expert.

Given this SQLite schema:
{schema}

Generate ONLY a valid SQLite SELECT query.
Do not explain.
Return only SQL.

User Question:
{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return clean_sql(response.text)