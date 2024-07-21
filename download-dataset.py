import os
import json
import logging
import time
import random
from sklearn.datasets import fetch_20newsgroups

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrapper


@log_time
def download_20newsgroups_subset(destination_folder='./20newsgroups', num_queries=25, num_docs_per_query=5):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    queries = {}
    ground_truth = {}
    collection = {}

    category_docs = {}
    for i, target in enumerate(newsgroups_data.target):
        if target not in category_docs:
            category_docs[target] = []
        category_docs[target].append(i)

    num_queries = min(num_queries, len(category_docs))
    selected_categories = random.sample(list(category_docs.keys()), num_queries)

    for i, category in enumerate(selected_categories):
        docs_for_category = random.sample(category_docs[category], num_docs_per_query)

        query_doc = newsgroups_data.data[docs_for_category[0]]
        query = ' '.join(query_doc.split()[:10])

        query_id = f'query_{i}'
        queries[query_id] = query
        ground_truth[query_id] = []

        for doc_index in docs_for_category:
            doc_id = f'doc_{doc_index}'
            collection[doc_id] = newsgroups_data.data[doc_index]
            ground_truth[query_id].append(doc_id)

    additional_docs = random.sample(range(len(newsgroups_data.data)), 1000)
    for doc_index in additional_docs:
        doc_id = f'doc_{doc_index}'
        if doc_id not in collection:
            collection[doc_id] = newsgroups_data.data[doc_index]

    with open(os.path.join(destination_folder, 'subset_queries.json'), 'w') as f:
        json.dump(queries, f)
    with open(os.path.join(destination_folder, 'subset_qrels.json'), 'w') as f:
        json.dump(ground_truth, f)
    with open(os.path.join(destination_folder, 'subset_collection.json'), 'w') as f:
        json.dump(collection, f)

    logging.info("Created and saved subset of 20 Newsgroups dataset.")


if __name__ == "__main__":
    download_20newsgroups_subset()
