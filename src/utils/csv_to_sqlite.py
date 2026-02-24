import pandas as pd
import sqlite3


CSV_PATH = "src/data/raw/customers-100000.csv"
DB_PATH = "src/data/raw/customers.db"
TABLE_NAME = "customers"


def convert_csv_to_db():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    print("Creating SQLite DB...")
    conn = sqlite3.connect(DB_PATH)

    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    conn.close()
    print("Database created successfully!")


if __name__ == "__main__":
    convert_csv_to_db()