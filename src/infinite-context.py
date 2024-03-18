import nltk
import numpy as np
import faiss
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from util import get_environment_variable

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class InfiniteContextRAG:
    def __init__(self, reference_data_path, similarity_threshold=0.8):
        self.reference_data_path = reference_data_path
        self.similarity_threshold = similarity_threshold
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.index = None
        self.levels = self.load_hierarchical_embeddings()

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()

        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Perform lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Join the tokens back into a string
        preprocessed_text = ' '.join(tokens)

        return preprocessed_text

    def extract_topics(self, text, num_topics=5):
        # Apply topic modeling techniques (e.g., LDA) to extract high-level topics
        lda = LatentDirichletAllocation(n_components=num_topics)
        topics = lda.fit_transform(text)
        return topics

    def generate_embeddings(self, text):
        # Generate contextual embeddings using BERT
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return embeddings

    def create_graph_structure(self, nodes, embeddings):
        # Create an empty adjacency matrix
        num_nodes = len(nodes)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))

        # Compute the cosine similarity between node embeddings
        similarity_matrix = cosine_similarity(embeddings)

        # Threshold the similarity matrix to create edges
        threshold = 0.5  # Adjust the threshold as needed
        adjacency_matrix = np.where(similarity_matrix >= threshold, similarity_matrix, 0)

        # Create a dictionary to represent the graph structure
        graph = {
            'nodes': nodes,
            'adjacency_matrix': adjacency_matrix
        }

        return graph

    def load_hierarchical_embeddings(self):
        text = self.preprocess_text(open(self.reference_data_path).read())
        topics = self.extract_topics(text)
        levels = [topics]

        current_level = topics
        while True:
            embeddings = self.generate_embeddings(current_level)
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.similarity_threshold)
            cluster_labels = clustering.fit_predict(embeddings)

            if len(np.unique(cluster_labels)) == 1:
                break

            current_level = [current_level[i] for i in range(len(current_level)) if cluster_labels[i] == 0]
            levels.append(current_level)

        graph = self.create_graph_structure(levels, embeddings)
        return levels, graph

    def generate_query_embedding(self, query):
        # Generate query embedding using BERT
        inputs = self.tokenizer(query, return_tensors='pt')
        outputs = self.model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return query_embedding

    def retrieve_documents(self, query_embedding, k=1):
        retrieved_docs = []
        current_level_docs = self.levels[0]  # Start with the top-level documents

        for level in self.levels[1:]:
            # Create or load the FAISS index for the current level
            if self.index is None:
                self.index = faiss.IndexFlatL2(query_embedding.shape[1])
            self.index.add(np.array(level).astype('float32'))

            # Retrieve top-k documents from the current level
            _, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
            current_level_docs = [level[i] for i in indices[0]]
            retrieved_docs.extend(current_level_docs)

        return retrieved_docs

    def generate_answer(self, prompt, max_length=100, num_return_sequences=1, temperature=0.7):
        # Load the GPT-2 tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # Generate the answer
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        # Decode the generated answer
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        return answer

    def rag_answer(self, query, k=2):
        query_embedding = self.generate_query_embedding(query)
        retrieved_docs = self.retrieve_documents(query_embedding, k)
        prompt = f"Based on: {', '.join(retrieved_docs)}. {query}"
        return self.generate_answer(prompt)


# Example usage
reference_data_path = 'infinite-context/reference/test-data.txt'
rag = InfiniteContextRAG(reference_data_path)
print(rag.rag_answer("Who led the Israelites out of Egypt?"))