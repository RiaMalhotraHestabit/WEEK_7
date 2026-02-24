import sqlite3


def load_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_info = ""

    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        schema_info += f"\nTable: {table_name}\nColumns:\n"
        for col in columns:
            schema_info += f"- {col[1]} ({col[2]})\n"

    conn.close()
    return schema_info