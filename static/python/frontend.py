import re
import numpy as np
import pandas as pd
import scipy
from spacy.lang.en import English
import itertools
from static.python.process_jobs import test_clean_posting, clean_title, zip_with_raw_titles
import string


def find_from_posting(job_posting: str, vectorizer, mask: np.array, model, filtered_df: pd.DataFrame,
                      column_name: str = 'KMeans'):
    vect = test_clean_posting(job_posting, vectorizer, mask)
    prediction, = model.predict(vect)
    return set(filtered_df[filtered_df[column_name] == prediction]["title"])


def find_from_posting_ordered(job_posting: str, cluster_data, column_name: str = 'KMeans'):
    vect = test_clean_posting(job_posting, cluster_data.vectorizer, cluster_data.mask)
    prediction, = cluster_data.model.predict(vect)
    temp_df = cluster_data.df.copy()
    temp_df['features'] = list(cluster_data.selected)
    temp_df = temp_df[temp_df[column_name] == prediction]
    title_stats = temp_df[['title', 'features']].groupby('title').sum()
    title_count = temp_df[['title', 'ID']].groupby('title').count()
    title_stats['mean'] = title_stats['features'] / title_count['ID']
    title_stats['distances'] = (title_stats['mean'] - vect).apply(
        scipy.sparse.linalg.norm)
    title_stats.sort_values(by='distances', axis=0, inplace=True)
    similar_titles = list(title_stats.index)
    return zip_with_raw_titles(similar_titles, temp_df)


def find_from_title(job, clusters):
    similar_jobs = set()
    clean_job = clean_title(job)
    for cluster in clusters:
        if clean_job in cluster:
            similar_jobs.update(cluster)
    return similar_jobs


def find_closest_posts(job, df):
    clean_job = clean_title(job)
    temp_df = df.copy()
    temp_df['features'] = list(cluster_data.selected)
    title_stats = temp_df[['title', 'features']].groupby('title').sum()
    title_count = temp_df[['title', 'ID']].groupby('title').count()
    title_stats['mean'] = title_stats['features'] / title_count['ID']
    temp_df['distances'] = (temp_df['features'] - title_stats.loc[clean_job, 'mean']).apply(
        scipy.sparse.linalg.norm)
    temp_df.sort_values(by='distances', axis=0, inplace=True)
    temp_df.groupby()
    idx = temp_df.groupby(['title'])['features'].max() == df['features']
    closest_postings = temp_df[idx][['title', 'raw_description']]
    return closest_postings


def reorder_titles_vertical(raw_data: iter):
    raw_data = list(raw_data)
    raw_data = raw_data[1:]
    length = len(raw_data)
    remainders = len(raw_data) % 4
    breaks = [len(raw_data) // 4 + 1 if i < remainders else len(raw_data) // 4 for i in range(4)]
    breaks = itertools.accumulate(breaks)
    breaks = list(breaks)
    breaks.insert(0, 0)
    raw_data = [[x for x in raw_data[breaks[i]:breaks[i + 1]]] for i in range(4)]
    data = [raw_data[i][j] for j in range(length // 4) for i in range(4)]
    for i in range(remainders):
        data.append(raw_data[i][length // 4])
    return data


def find_from_title_ordered(job, cluster_data, cluster_name='KMeans'):
    num_clusters = cluster_data.model.cluster_centers_.shape[0]
    temp_df = cluster_data.df.copy()
    temp_df['features'] = list(cluster_data.selected)
    clean_job = clean_title(job)
    cluster_indices = []
    meta_cluster = [set(temp_df[temp_df[cluster_name] == x]["title"]) for x in
                    range(num_clusters)]
    for i, cluster in enumerate(meta_cluster):
        if clean_job in cluster:
            cluster_indices.append(i)
    if not cluster_indices:
        # Checks to see if the title exists in the database
        return None
    temp_df = temp_df[temp_df[cluster_name].apply(lambda x: x in cluster_indices)]
    title_stats = temp_df[['title', 'features']].groupby('title').sum()
    title_count = temp_df[['title', 'ID']].groupby('title').count()
    title_stats['mean'] = title_stats['features'] / title_count['ID']
    title_stats['distances'] = (title_stats['mean'] - title_stats.loc[clean_job, 'mean']).apply(
        scipy.sparse.linalg.norm)
    title_stats.sort_values(by='distances', axis=0, inplace=True)
    similar_jobs = list(title_stats.index)
    return zip_with_raw_titles(similar_jobs, temp_df)


def compute_medoids(df: pd.DataFrame, selected: np.array, clustering: str = "KMeans") -> set:
    num_clusters = df[clustering].nunique()
    medoids = set()
    temp_df = df
    temp_df['features'] = list(selected)
    for i in range(num_clusters):
        cluster = list(temp_df[temp_df[clustering] == i]['features'])
        titles = list(temp_df[temp_df[clustering] == i]["title"])
        mean = np.zeros([1, selected.shape[1]])
        for x in cluster:
            mean += x
        mean /= len(cluster)
        distances = [scipy.linalg.norm((mean - x)) for x in cluster]
        medoid_index = distances.index(min(distances))
        medoids.add(titles[medoid_index])
    return zip_with_raw_titles(medoids, df)


def highlight_text(text: str, special_words) -> str:
    text = re.sub('\s+', ' ', text).strip()
    nlp = English()
    string_list = []
    for word in text.split(' '):
        stem = word
        stem = stem.translate(str.maketrans('', '', string.punctuation))
        stem = nlp(stem.strip().upper().lower())
        stem = stem[0].lemma_ if stem else stem
        if stem in special_words or word in special_words:
            string_list.append(f'<span style="background-color: #82f9ff">{word}</span>')
        else:
            string_list.append(word)
    return ' '.join(string_list)


# if __name__ == '__main__':
#     from joblib import load
#
#     cluster_data = load('data/nyc_clustered.pkl')
#     COMPONENTS = 4
#     trans_data = manifold.Isomap(n_neighbors=5, n_components=COMPONENTS, n_jobs=-1).fit_transform(
#         cluster_data.selected.toarray())
#
#     plt.figure(figsize=(12, 8))
#     plt.title('Decomposition using ISOMAP')
#     for i in range(COMPONENTS - 1):
#         plt.scatter(trans_data[:, i], trans_data[:, i + 1])
#     plt.show()
