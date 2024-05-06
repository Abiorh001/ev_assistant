import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url

connection_string = "postgresql://postgres:Lucifer_001@localhost:5432"
db_name = "ocpp_vector"
conn = psycopg2.connect(connection_string)
conn.autocommit = True

try:
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    exists = bool(cursor.fetchone())
    cursor.close()

    if exists:
        print("Database exists")
        vector_store = PGVectorStore.from_params(
            database="ocpp_vector",
            host="localhost",
            password="Lucifer_001",
            port=5432,
            user="postgres",
            table_name="ocpp_data",
            embed_dim=1536,
        )
    else:
        print("Database does not exist")
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE {db_name}")
        cursor.close()
        print("Database created")

        # Now that the database is created, let's connect to it
        url = make_url(connection_string)
        vector_store = PGVectorStore.from_params(
            database=db_name,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name="ocpp_data",
            embed_dim=1536,
        )

except psycopg2.Error as e:
    print(f"Error: {e}")

