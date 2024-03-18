import numpy as np
import faiss
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import NMF
from transformers import T5ForConditionalGeneration, T5Tokenizer
from util import get_environment_variable

nltk.download('punkt')
nltk.download('stopwords')

class InfiniteContextRAG:
    def __init__(self, reference_data_path, similarity_threshold=0.8):
        print("Initializing InfiniteContextRAG...")
        self.reference_data_path = reference_data_path
        self.similarity_threshold = similarity_threshold
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.index = None
        self.levels = []
        self.graph = None
        self.initialize_system()
        print("InfiniteContextRAG initialized.")

    def preprocess_text(self, text):
        print("Preprocessing text...")
        tokens = self.tokenizer.tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        preprocessed_text = ' '.join(tokens)
        print("Text preprocessed.")
        return preprocessed_text

    def extract_topics(self, documents, num_topics=5):
        print("Extracting topics...")
        embeddings = self.generate_embeddings(documents)
        embeddings = np.abs(embeddings)  # Take the absolute values of the embeddings
        nmf = NMF(n_components=num_topics)
        topics = nmf.fit_transform(embeddings)
        print("Topics extracted.")
        return topics

    def generate_embeddings(self, documents, segment_size=512):
        print("Generating embeddings...")
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
        print("Creating graph structure...")
        num_nodes = len(nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        similarity_matrix = cosine_similarity(embeddings)
        threshold = 0.5
        adjacency_matrix = np.where(similarity_matrix >= threshold, similarity_matrix, 0)
        graph = {'nodes': nodes, 'adjacency_matrix': adjacency_matrix}
        print("Graph structure created.")
        return graph

    def load_hierarchical_embeddings(self):
        print("Loading hierarchical embeddings...")
        text = self.preprocess_text(open(self.reference_data_path).read())
        topics = self.extract_topics(text)
        levels = [topics]

        current_level = topics
        while True:
            print(f"Processing level {len(levels)}...")
            embeddings = self.generate_embeddings(current_level)
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.similarity_threshold)
            cluster_labels = clustering.fit_predict(embeddings)

            if len(np.unique(cluster_labels)) == 1:
                break

            current_level = [current_level[i] for i in range(len(current_level)) if cluster_labels[i] == 0]
            levels.append(current_level)

        graph = self.create_graph_structure(levels, embeddings)
        print("Hierarchical embeddings loaded.")
        return levels, graph

    def generate_query_embedding(self, query):
        print("Generating query embedding...")
        inputs = self.tokenizer(query, return_tensors='pt')
        outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        print("Query embedding generated.")
        return query_embedding

    def retrieve_documents(self, query_embedding, k=1):
        print("Retrieving documents...")
        retrieved_docs = []
        current_level_docs = self.levels[0]

        for level in self.levels[1:]:
            if self.index is None:
                self.index = faiss.IndexFlatL2(query_embedding.shape[1])
            self.index.add(np.array(level).astype('float32'))

            _, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
            current_level_docs = [level[i] for i in indices[0]]
            retrieved_docs.extend(current_level_docs)

        print("Documents retrieved.")
        return retrieved_docs

    def generate_answer(self, prompt):
        print("Generating answer...")
        input_ids = self.t5_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.t5_model.generate(input_ids)
        answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Answer generated.")
        return answer

    def rag_answer(self, query, k=2):
        print("Processing query:", query)
        query_embedding = self.generate_query_embedding(query)
        retrieved_docs = self.retrieve_documents(query_embedding, k)
        prompt = f"Based on: {', '.join(retrieved_docs)}. {query}"
        answer = self.generate_answer(prompt)
        print("RAG answer:", answer)
        return answer

    def initialize_system(self):
        self.levels, self.graph = self.load_hierarchical_embeddings()


# Example usage
reference_data_path = '../reference/test-data.txt'
rag = InfiniteContextRAG(reference_data_path)
print(rag.rag_answer("Who led the Israelites out of Egypt?"))