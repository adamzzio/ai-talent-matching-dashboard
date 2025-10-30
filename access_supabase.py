import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("host"),
    port=os.getenv("port"),
    dbname=os.getenv("database"),
    user=os.getenv("user"),
    password=os.getenv("password"),
    sslmode="require",
    connect_timeout=20,
)

with conn, conn.cursor() as cur:
    cur.execute("""
        SELECT nspname AS schema_name
        FROM pg_catalog.pg_namespace
        WHERE nspname NOT LIKE 'pg_%'
          AND nspname <> 'information_schema'
        ORDER BY nspname;
    """)
    rows = cur.fetchall()
    print("Schemas:")
    for (schema,) in rows:
        print(" -", schema)
