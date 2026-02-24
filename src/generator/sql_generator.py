import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)

MODEL_NAME = "models/gemini-2.0-flash-lite"   


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

    sql = response.text.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql