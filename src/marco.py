"""
MARCO: Multi-layered Abstraction Retrieval and Contextual Answering

This module implements the MARCO system, which utilizes a multi-layered abstraction approach to efficiently retrieve relevant information from a large corpus of documents and generate accurate and contextually appropriate answers. The system mimics the way humans process and recall information by leveraging techniques like summarization, topic modeling, and graph-based representation.

Author: Alan Hourmand"""

import numpy as np
import faiss
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from util import get_environment_variable
from gpt_handler import GPTHandler

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords')


class MARCO:
    def __init__(self, reference_path, language_model, similarity_threshold=0.8, satisfaction_threshold=0.7):
        print("Initializing InfiniteContextRAG...")

        self.language_model = language_model
        self.reference_data_path = reference_path
        self.similarity_threshold = similarity_threshold
        self.satisfaction_threshold = satisfaction_threshold
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.index = None
        self.levels = []
        self.notes = []
        self.weighted_chunks = {}
        self.graph = None
        self.initialize_system()

        print("InfiniteContextRAG initialized.")

    def preprocess_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    def extract_topics(self, documents, num_topics=None):
        preprocessed_docs = [self.preprocess_text(doc) for doc in documents]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        topics = lda.fit_transform(tfidf_matrix)
        return topics

    def generate_embeddings(self, documents, segment_size=512):
        embeddings = []
        total_segments = sum(len(doc) // segment_size + 1 for doc in documents)
        segment_counter = 0

        for doc in documents:
            if isinstance(doc, str):
                doc_embeddings = []
                for i in range(0, len(doc), segment_size):
                    segment = doc[i:i + segment_size]
                    inputs = self.tokenizer(segment, padding=True, truncation=True, return_tensors='pt')
                    outputs = self.model(**inputs)
                    segment_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
                    doc_embeddings.append(segment_embeddings)

                    segment_counter += 1
                    progress = (segment_counter / total_segments) * 100
                    print(f"Embedding generation progress: {progress:.2f}%", end='\r')

                doc_embeddings = np.concatenate(doc_embeddings, axis=0)
                embeddings.append(doc_embeddings)
            elif isinstance(doc, list):
                doc_embeddings = self.generate_embeddings(doc, segment_size)
                embeddings.extend(doc_embeddings)
            else:
                raise ValueError("Unsupported document type")

        embeddings = np.concatenate(embeddings, axis=0)
        print("\nEmbeddings generated.")
        return embeddings

    def create_graph_structure(self, nodes, embeddings):
        num_nodes = len(nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        similarity_matrix = cosine_similarity(embeddings)
        adjacency_matrix = np.where(similarity_matrix >= self.similarity_threshold, similarity_matrix, 0)
        graph = {'nodes': nodes, 'adjacency_matrix': adjacency_matrix}
        return graph

    def load_hierarchical_embeddings(self):
        text = self.preprocess_text(open(self.reference_data_path).read())
        self.levels = self.create_abstraction_layers(text)
        self.graph = self.create_graph_structure(self.levels, self.generate_embeddings([layer['summary'] for layer in self.levels]))

    def generate_query_embedding(self, query):
        inputs = self.tokenizer(query, return_tensors='pt')
        outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return query_embedding

    def retrieve_documents(self, query_embedding, layer_embeddings, k=1):
        similarity_scores = cosine_similarity(query_embedding, layer_embeddings)[0]
        top_indices = similarity_scores.argsort()[-k:][::-1]
        retrieved_docs = [layer_embeddings[i] for i in top_indices]
        return retrieved_docs

    def extract_relevant_info(self, retrieved_docs, query):
        relevant_info = []
        layer_notes = []
        query_tokens = set(self.tokenizer.tokenize(query))

        for doc in retrieved_docs:
            doc_tokens = set(self.tokenizer.tokenize(doc))
            if len(query_tokens.intersection(doc_tokens)) > 0:
                relevant_info.append(doc)
                notes = self.extract_notes(doc, query)
                layer_notes.extend(notes)

        return relevant_info, layer_notes

    def update_chunk_weights(self, query, relevant_info, notes):
        for info, note in zip(relevant_info, notes):
            chunk = self.generate_chunk(info)
            relevance_score = self.calculate_relevance_score(query, info, note)
            self.weighted_chunks[chunk] = relevance_score

    def generate_chunk(self, text, chunk_size=100):
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    def calculate_relevance_score(self, query, info, note):
        # Calculate cosine similarity between query and chunk embeddings
        query_embedding = self.generate_query_embedding(query)
        chunk_embedding = self.generate_embeddings([info])[0]
        cosine_sim = cosine_similarity([query_embedding], [chunk_embedding])[0][0]

        # Calculate TF-IDF scores for words in the chunk
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([info])
        tfidf_scores = tfidf_matrix.toarray()[0]
        tfidf_score = np.mean(tfidf_scores)

        # Combine cosine similarity and TF-IDF score
        relevance_score = 0.6 * cosine_sim + 0.4 * tfidf_score
        return relevance_score

    def generate_contextual_answer(self, query, relevant_info, notes, weighted_chunks):
        prompt = f"Query: {query}\n\nRelevant Information:\n{relevant_info}\n\nNotes:\n{notes}\n\nWeighted Chunks:\n{weighted_chunks}\n\nAnswer:"
        answer = self.language_model.generate_response(prompt)
        return answer

    def rag_answer(self, query, k=2, max_iterations=5):
        print("Processing query:", query)
        self.notes = []
        self.weighted_chunks = {}

        relevant_layers = self.determine_relevant_layers(query)

        iteration = 0
        satisfaction_score = 0.0

        while satisfaction_score < self.satisfaction_threshold and iteration < max_iterations:
            print(f"Iteration {iteration + 1}")

            for layer in relevant_layers:
                query_embedding = self.generate_query_embedding(query)
                retrieved_docs = self.retrieve_documents(query_embedding, layer['embeddings'], k)
                relevant_info, layer_notes = self.extract_relevant_info(retrieved_docs, query)
                self.notes.extend(layer_notes)

                self.update_chunk_weights(query, relevant_info, layer_notes)

                if self.need_more_info(query, relevant_info):
                    additional_docs = self.refine_search(query, layer)
                    additional_info, additional_notes = self.extract_relevant_info(additional_docs, query)
                    relevant_info.extend(additional_info)
                    self.notes.extend(additional_notes)

                    self.update_chunk_weights(query, additional_info, additional_notes)

            satisfaction_score = self.calculate_satisfaction_score(relevant_info, self.notes, self.weighted_chunks)
            print(f"Satisfaction Score: {satisfaction_score}")

            if satisfaction_score >= self.satisfaction_threshold:
                break

            iteration += 1

        if satisfaction_score >= self.satisfaction_threshold:
            print("Satisfactory information obtained.")
        else:
            print("Maximum iterations reached. Generating answer with available information.")

        answer = self.generate_contextual_answer(query, relevant_info, self.notes, self.weighted_chunks)
        print("RAG answer:", answer)
        return answer

    def calculate_satisfaction_score(self, relevant_info, notes, weighted_chunks):
        # Calculate average relevance score
        relevance_scores = [score for chunk, score in weighted_chunks.items() if chunk in relevant_info]
        avg_relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        # Calculate coverage of query topics
        query_topics = set(self.extract_topics([self.preprocess_text(note) for note in notes], num_topics=3))
        coverage_score = len(query_topics.intersection(set(weighted_chunks.keys()))) / len(query_topics)

        # Combine average relevance score and coverage score
        satisfaction_score = 0.7 * avg_relevance_score + 0.3 * coverage_score
        return satisfaction_score

    def determine_relevant_layers(self, query):
        query_embedding = self.generate_query_embedding(query)
        similarity_scores = [cosine_similarity([query_embedding], [layer['embeddings']])[0][0] for layer in self.levels]
        top_k_layers = [self.levels[i] for i in np.argsort(similarity_scores)[::-1][:2]]
        return top_k_layers

    def need_more_info(self, query, relevant_info):
        satisfaction_score = self.calculate_satisfaction_score(relevant_info, self.notes, self.weighted_chunks)
        return satisfaction_score < self.satisfaction_threshold

    def refine_search(self, query, layer):
        # Expand search to neighboring layers
        layer_index = self.levels.index(layer)
        neighboring_layers = self.levels[max(0, layer_index - 1):min(len(self.levels), layer_index + 2)]
        neighboring_docs = [doc for layer in neighboring_layers for doc in layer['summary']]

        # Perform focused search within the neighboring layers
        query_embedding = self.generate_query_embedding(query)
        retrieved_docs = self.retrieve_documents(query_embedding, neighboring_docs, k=2)
        return retrieved_docs

    def interactive_mode(self):
        while True:
            user_input = input("Ask a question (or type 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
            answer = self.rag_answer(user_input)
            print("RAG answer:", answer)

    def initialize_system(self):
        self.load_hierarchical_embeddings()

    def create_abstraction_layers(self, text, num_layers=3):
        layers = []
        current_text = text

        for i in range(num_layers):
            summary = self.summarize_text(current_text)
            topics = self.extract_topics(summary)
            embeddings = self.generate_embeddings(summary)
            layer = {
                'summary': summary,
                'topics': topics,
                'embeddings': embeddings
            }
            layers.append(layer)
            current_text = summary

        return layers

    def summarize_text(self, text):
        return self.language_model.generate_summary(text)

    def generate_embeddings(self, text):
        return self.language_model.generate_embeddings(text)

    def extract_notes(self, text, query):
        notes = []
        sentences = nltk.sent_tokenize(text)
        query_tokens = set(self.tokenizer.tokenize(query))

        for sentence in sentences:
            sentence_tokens = set(self.tokenizer.tokenize(sentence))
            if len(query_tokens.intersection(sentence_tokens)) > 0:
                notes.append(sentence)

        return notes


# Example usage
language_model = GPTHandler()
reference_data_path = '../reference/test-data.txt'
rag = MARCO(reference_data_path, language_model)
rag.interactive_mode()