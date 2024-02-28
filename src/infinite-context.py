# infinite-context.py
import util
from gpt_handler import GPTHandler
import psycopg2
import numpy as np
from psycopg2.extras import execute_values

# Retrieve database connection string from environment variables
DB_CONNECTION_STRING = util.get_environment_variable('DB_CONNECTION_STRING')

# Instantiate GPTHandler with your chosen model
gpt = GPTHandler(model="gpt-3.5-turbo-1106")

# Connect to the PostgreSQL database
conn = psycopg2.connect(DB_CONNECTION_STRING)
cur = conn.cursor()

# Enable pgvector extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# Define the table schema for storing text embeddings
cur.execute("""
CREATE TABLE IF NOT EXISTS text_embeddings (
    id SERIAL PRIMARY KEY,
    document TEXT NOT NULL,
    embedding vector(512)
);
""")
conn.commit()


def generate_embedding(text):
    """
    Generate a text embedding using the GPT model.
    """
    # Generate the embedding using the GPTHandler instance or another method
    # This is a placeholder, replace with actual method to obtain embeddings
    embedding = gpt.create_embedding(text)
    return embedding


def store_embedding_in_db(document, embedding):
    """
    Store the document and its embedding in the database.
    """
    # Insert the document and its embedding into the database
    cur.execute("INSERT INTO text_embeddings (document, embedding) VALUES (%s, %s)",
                (document, embedding))
    conn.commit()


def batch_store_embeddings_in_db(data):
    """
    Batch store documents and their embeddings in the database for efficiency.
    """
    # Prepare the list of tuples to insert
    embeddings_data = [(text, generate_embedding(text)) for text in data]
    # Use execute_values to perform batch insertion
    execute_values(cur,
                   "INSERT INTO text_embeddings (document, embedding) VALUES %s",
                   embeddings_data)
    conn.commit()


# Example usage of the functions
# Load your data - this should be a list of documents/texts
with open('reference/test-data.txt', 'r') as file:
    data = file.readlines()

# Batch store documents and their embeddings in the database
batch_store_embeddings_in_db(data)

# Don't forget to close the cursor and connection when done
cur.close()
conn.close()
