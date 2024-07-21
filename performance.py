import os
import re
import nltk
import time
import logging
import json
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from spellchecker import SpellChecker
from nltk.corpus import wordnet
from itertools import chain
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"{func.__name__} took {execution_time:.2f} seconds")
        return result

    return wrapper


@log_time
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)


@log_time
def load_data(data_folder='./20newsgroups'):
    with open(os.path.join(data_folder, 'subset_queries.json'), 'r') as f:
        queries = json.load(f)
    with open(os.path.join(data_folder, 'subset_qrels.json'), 'r') as f:
        ground_truth = json.load(f)
    with open(os.path.join(data_folder, 'subset_collection.json'), 'r') as f:
        collection = json.load(f)
    return queries, ground_truth, collection


@log_time
def preprocess_data(collection):
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.WordNetLemmatizer()

    def preprocess_text(text):
        text = re.sub(r'\W+', ' ', text.lower())
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_set]
        return ' '.join(tokens)

    processed_data = {doc_id: preprocess_text(text) for doc_id, text in collection.items()}
    return processed_data


@log_time
def create_tfidf_vectorizer(processed_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(processed_data.values())
    return vectorizer, X


@log_time
def create_sentence_transformer(processed_data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = model.encode(list(processed_data.values()), convert_to_tensor=True)
    return model, corpus_embeddings


class RetrievalMethods:
    def __init__(self, vectorizer, X, model, corpus_embeddings, collection):
        self.vectorizer = vectorizer
        self.X = X
        self.model = model
        self.corpus_embeddings = corpus_embeddings
        self.collection = collection
        self.spell = SpellChecker()
        self.stopwords_set = set(nltk.corpus.stopwords.words('english'))

    def preprocess_text(self, text):
        text = re.sub(r'\W+', ' ', text.lower())
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stopwords_set]
        return ' '.join(tokens)

    def correct_spelling(self, text):
        return ' '.join([self.spell.correction(word) or word for word in text.split()])

    def dynamic_threshold(self, scores, base_threshold=0.05, min_docs=2, max_docs=10):
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) < min_docs:
            return 0
        elif len(sorted_scores) <= max_docs:
            return sorted_scores[-1]
        else:
            return max(base_threshold, sorted_scores[max_docs - 1])

    @log_time
    def tfidf_retrieval(self, query, top_k=10):
        query_vec = self.vectorizer.transform([self.preprocess_text(query)])
        scores = (self.X * query_vec.T).toarray().flatten()
        threshold = self.dynamic_threshold(scores)
        ranked_indices = scores.argsort()[::-1]

        retrieved_docs = []
        for i in range(min(len(ranked_indices), top_k)):
            if scores[ranked_indices[i]] > threshold:
                retrieved_docs.append((list(self.collection.keys())[ranked_indices[i]], scores[ranked_indices[i]]))

        return retrieved_docs

    @log_time
    def dense_retrieval(self, query, top_k=10):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings)[0]
        scores_np = scores.cpu().numpy()
        threshold = self.dynamic_threshold(scores_np)
        ranked_indices = scores.argsort(descending=True)

        retrieved_docs = []
        for i in range(min(len(ranked_indices), top_k)):
            if scores[ranked_indices[i]] > threshold:
                retrieved_docs.append(
                    (list(self.collection.keys())[ranked_indices[i]], scores[ranked_indices[i]].item()))

        return retrieved_docs

    @log_time
    def spelling_correction_retrieval(self, query, top_k=10):
        corrected_query = self.correct_spelling(query)
        return self.tfidf_retrieval(corrected_query, top_k)

    @log_time
    def synonyms_retrieval(self, query, top_k=10):
        expanded_query = self.expand_with_synonyms(query)
        return self.tfidf_retrieval(expanded_query, top_k)

    def expand_with_synonyms(self, query):
        words = query.split()
        expanded_query = []
        for word in words:
            synonyms = wordnet.synsets(word)
            lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
            lemmas.add(word)
            expanded_query.extend(list(lemmas)[:3])
        return ' '.join(expanded_query)


@log_time
def mean_reciprocal_rank(y_true_list):
    rr_sum = sum(1 / (y.index(1) + 1) if 1 in y else 0 for y in y_true_list)
    return rr_sum / len(y_true_list)


def mean_average_precision(y_true, y_score, k):
    average_precisions = []
    for i in range(len(y_true)):
        relevant_docs = 0
        precision_at_i = []
        for j in range(len(y_true[i])):
            if y_true[i][j] == 1:
                relevant_docs += 1
                precision_at_i.append(relevant_docs / (j + 1))
        if len(precision_at_i) == 0:
            average_precisions.append(0)
        else:
            average_precisions.append(sum(precision_at_i) / min(k, len(y_true[i])))
    return sum(average_precisions) / len(average_precisions)


@log_time
def evaluate_retrieval_method(method, queries, ground_truth, collection, top_k=10):
    precision_scores, recall_scores, f1_scores, ndcg_scores, map_scores = [], [], [], [], []
    y_true_list = []
    execution_times = []

    for query_id, query in queries.items():
        truth = set(ground_truth[query_id])
        start_time = time.time()
        retrieved = method(query, top_k=top_k)
        end_time = time.time()
        execution_times.append(end_time - start_time)

        if len(retrieved) < 2:  # Skip if fewer than 2 documents are retrieved
            continue

        y_true = [1 if doc_id in truth else 0 for doc_id, _ in retrieved]
        y_score = [score for _, score in retrieved]

        true_positives = sum(y_true)
        precision = true_positives / len(retrieved) if retrieved else 0
        recall = true_positives / len(truth) if truth else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        ideal_dcg = ndcg_score([[1] * len(truth)], [[1] * len(truth)])
        actual_ndcg = ndcg_score([y_true], [y_score])
        ndcg = actual_ndcg / ideal_dcg if ideal_dcg > 0 else 0

        ap = sum((sum(y_true[:i + 1]) / (i + 1)) * y for i, y in enumerate(y_true)) / len(truth) if truth else 0

        y_true_list.append(y_true)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ndcg_scores.append(ndcg)
        map_scores.append(ap)

        print(f"Query: {query[:50]}...")
        print(f"Ground truth: {truth}")
        print(f"Retrieved: {[doc_id for doc_id, _ in retrieved[:5]]}...")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, NDCG: {ndcg:.4f}, AP: {ap:.4f}")
        print("----")

    if not y_true_list:
        logging.warning(f"No valid results for method {method.__name__}")
        return {metric: 0 for metric in ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr', 'avg_time']}

    mrr = mean_reciprocal_rank(y_true_list)

    return {
        'precision': sum(precision_scores) / len(precision_scores),
        'recall': sum(recall_scores) / len(recall_scores),
        'f1': sum(f1_scores) / len(f1_scores),
        'ndcg': sum(ndcg_scores) / len(ndcg_scores),
        'map': sum(map_scores) / len(map_scores),
        'mrr': mrr,
        'avg_time (seconds)': sum(execution_times) / len(execution_times)
    }


@log_time
def performance_testing():
    logging.info("Starting the information retrieval benchmark")

    download_nltk_data()

    queries, ground_truth, collection = load_data()
    processed_data = preprocess_data(collection)
    vectorizer, X = create_tfidf_vectorizer(processed_data)
    model, corpus_embeddings = create_sentence_transformer(processed_data)

    retrieval_methods = RetrievalMethods(vectorizer, X, model, corpus_embeddings, collection)

    methods = {
        'TF-IDF': retrieval_methods.tfidf_retrieval,
        'Dense Retrieval': retrieval_methods.dense_retrieval,
        'Spelling Correction': retrieval_methods.spelling_correction_retrieval,
        'Synonyms Handling': retrieval_methods.synonyms_retrieval
    }

    results = {}
    for name, method in methods.items():
        logging.info(f"Evaluating {name}")
        results[name] = evaluate_retrieval_method(method, queries, ground_truth, collection, top_k=10)

    # Print results
    for method, metrics in results.items():
        logging.info(f"Results for {method}: {metrics}")

    # Visualization
    plt.figure(figsize=(16, 12))
    x = range(len(methods))
    metrics = ['precision', 'recall', 'f1', 'ndcg', 'map', 'mrr', 'avg_time (seconds)']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'cyan']

    for i, metric in enumerate(metrics):
        values = [results[method][metric] for method in methods]
        plt.bar([xi + i * 0.1 for xi in x], values, width=0.1, label=metric.upper(), color=colors[i])

    plt.xlabel('Retrieval Methods')
    plt.ylabel('Scores')
    plt.title('Comparison of Retrieval Methods')
    plt.xticks([xi + 0.3 for xi in x], methods.keys())
    plt.legend()

    # Save the plot
    plt.savefig('retrieval_methods_comparison.png')
    logging.info("Saved comparison plot as 'retrieval_methods_comparison.png'")


if __name__ == "__main__":
    performance_testing()
