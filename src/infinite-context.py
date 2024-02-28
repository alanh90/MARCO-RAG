# infinite-context.py
from util import load_env_var
from gpt_handler import GPTHandler
import psycopg2
from pgvector import register_vector

# Set up your database connection string from environment variables
TIMESCALE_CONNECTION_STRING = load_env_var('TIMESCALE_CONNECTION_STRING')

# Instantiate GPTHandler with your chosen model
gpt = GPTHandler(model="gpt-3.5-turbo-1106")

# Connect to the PostgreSQL database
conn = psycopg2.connect(TIMESCALE_CONNECTION_STRING)
register_vector(conn)
cur = conn.cursor()

# Define the table schema for the summaries
cur.execute("""
CREATE TABLE IF NOT EXISTS summaries (
    id BIGSERIAL PRIMARY KEY,
    title TEXT,
    summary TEXT,
    details TEXT,
    embedding VECTOR(2048)
);
""")
conn.commit()

def generate_summary(text):
    """
    Generate a summary of the given text using the GPT model.
    """
    # Use the GPTHandler instance to generate a summary
    summary = gpt.generate_response(text)
    return summary.strip()

def generate_details(text):
    """
    Generate detailed points from the text using the GPT model.
    """
    # Use the GPTHandler instance to generate details
    details = gpt.generate_response(text, specific_parameters_for_details)
    return details.strip()

def store_summary_in_db(title, summary, details, embedding):
    """
    Store the generated summary, details, and embedding in the database.
    """
    # Insert the summary and details into the summaries table
    cur.execute("INSERT INTO summaries (title, summary, details, embedding) VALUES (%s, %s, %s, %s)",
                (title, summary, details, embedding))
    conn.commit()

def create_embedding_for_text(text):
    """
    Create an embedding vector for the given text.
    """
    # Use the GPTHandler instance or another method to generate an embedding for the text
    embedding = gpt.create_embedding(text)
    return embedding

# Example usage of the functions
text_to_summarize = "Your large text dataset here"
summary = generate_summary(text_to_summarize)
details = generate_details(text_to_summarize)
embedding = create_embedding_for_text(text_to_summarize)

# Store the generated data in the database
store_summary_in_db("Title of the text", summary, details, embedding)

# Close the cursor and connection when done
cur.close()
conn.close()
