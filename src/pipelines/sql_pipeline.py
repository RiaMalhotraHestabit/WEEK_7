import os
import re
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from google import genai
from src.utils.schema_loader import load_schema
from src.generator.sql_generator import generate_sql

load_dotenv()

DB_PATH = "src/data/raw/customers.db"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.5-flash"

def validate_sql(sql: str) -> bool:
    sql = sql.strip().lower()

    # Must start with SELECT
    if not re.match(r"^select\b", sql):
        return False

    # Block dangerous operations
    forbidden_keywords = ["drop", "delete", "insert", "update", "alter"]
    if any(word in sql for word in forbidden_keywords):
        return False

    return True


def execute_sql(sql: str):
    conn = sqlite3.connect(DB_PATH)

    try:
        cursor = conn.cursor()
        cursor.execute(sql)

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        rows = rows[:100]

        return columns, rows

    finally:
        conn.close()


def summarize_result(question: str, columns, rows):
    table_preview = f"Columns: {columns}\nRows: {rows}"

    prompt = f"""
User Question: {question}

SQL Result:
{table_preview}

Provide a short natural language summary.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text.strip()

#Main pipeline
def run():
    print("\nSQL-QA Engine Started\n")

    question = input("Ask your database a question: ")

    print("\nLoading schema...")
    schema = load_schema(DB_PATH)

    print("\nGenerating SQL...")
    sql = generate_sql(question, schema)

    print("\nGenerated SQL:")
    print(sql)

    if not validate_sql(sql):
        print("\nUnsafe SQL detected. Aborting.")
        return

    print("\nExecuting SQL...")
    columns, rows = execute_sql(sql)

    print("\nRaw Results:")
    for row in rows:
        print(row)

    print("\nSummarizing...")
    summary = summarize_result(question, columns, rows)

    print("\nFinal Answer:")
    print(summary)

#---for capstone helper----

def query_sql(question):
    """
    Capstone helper for /ask-sql
    Returns: {"answer": str, "confidence": float}
    """
    schema = load_schema(DB_PATH)
    sql = generate_sql(question, schema)

    if not validate_sql(sql):
        return {"answer": "Unsafe SQL detected. Cannot execute.", "confidence": 0.0}

    columns, rows = execute_sql(sql)
    if rows:
    # Create a pretty table using pandas
        df = pd.DataFrame(rows, columns=columns)
        table_preview = df.to_string(index=False)
    else:
        table_preview = "No results found."

    summary = summarize_result(question, columns, rows)

    # Simple confidence heuristic: if fewer than 5 rows, reduce confidence
    confidence = 0.9 if len(rows) >= 5 else 0.7

    return {"Summary": summary, "answer": f"{table_preview}\n\n", "confidence": confidence}

if __name__ == "__main__":
    run()
