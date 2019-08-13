import dill
import pandas as pd
import os
from static.python.process_jobs import clean_encoding, process_jobs, feature_selection, add_kmeans_labels, ClusterData,\
    add_gmm_labels
import logging

logging.basicConfig(level=logging.INFO, filename='temp/dump.log',
                    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')

PROCESS_REFRESH = False

NYC_CLUSTER_PATHS = ("temp/nyc_selected.pkl", 'temp/nyc_clustered.pkl', 'temp/nyc_relevant_words.pkl')
ARME_CLUSTER_PATHS = ("temp/arme_selected.pkl", 'temp/arme_clustered.pkl', 'temp/arme_relevant_words.pkl')


def process_filter_dump(df, inv_reg: float = 3, penalty: str = 'l1', bootstrap: int = 0,
                        proportion: float = 0.8, refresh=False, selected_path='data/dump.pkl', name='nyc'):
    tfidf, titles, vectorizer, df_filtered = process_jobs(df, refresh=PROCESS_REFRESH, name=name)
    print('Now doing feature selection...')
    selected, mask = feature_selection(tfidf, titles, inv_reg=inv_reg, penalty=penalty, name=name, bootstrap=bootstrap,
                                       proportion=proportion, refresh=refresh)
    with open(selected_path, "wb") as f:
        dill.dump((selected, vectorizer, mask, df_filtered), f)


def cluster_dump(cluster_alg: str = 'KMeans', input_path="temp/nyc_selected.pkl",
                 output_path='temp/nyc_clustered.pkl', relevant_path='temp/nyc_relevant_words.pkl', dynamic=False,
                 min_size=0, max_size=float('inf'), max_cycles=5, k=60):
    with open(input_path, 'rb') as f:
        selected, vectorizer, mask, df = dill.load(f)
    print('Now clustering...')
    if cluster_alg == 'KMeans':
        model = add_kmeans_labels(df, selected, k, dynamic, min_size, max_size, max_cycles)
    elif cluster_alg == 'GMM':
        model = add_gmm_labels(df, selected, k)
    else:
        raise Exception(f"Invalid clustering algorithm: {cluster_alg}")

    cluster_data = ClusterData(selected, vectorizer, mask, model, df)
    with open(output_path, 'wb') as f:
        dill.dump(cluster_data, f)

    relevant_words = [cluster_data.vectorizer.get_feature_names()[i] for i, unfiltered in
                      enumerate(cluster_data.mask) if cluster_data.mask[i]]
    with open(relevant_path, 'wb') as f2:
        dill.dump(relevant_words, f2)


def nyc_process_filter_dump(inv_reg: float = 3, penalty: str = 'l1', bootstrap: int = 0,
                            proportion: float = 0.8, refresh=False) -> None:
    selected_path = "temp/nyc_selected.pkl"
    df = pd.read_csv('temp/NYC_Jobs.csv', encoding='utf-8')
    df['description'] = df['Job Description'].astype('str') + " " + \
                        df['Minimum Qual Requirements'].astype('str') + " " + \
                        df['Preferred Skills'].astype('str')
    df['description'] = df['description'].apply(clean_encoding)
    df = df.rename(columns={'Business Title': 'title', 'Job ID': 'ID'})
    df = df.filter(['ID', 'title', 'description'])
    process_filter_dump(df, inv_reg, penalty, bootstrap, proportion, refresh, selected_path, name='nyc')


def arme_process_filter_dump(inv_reg: float = 3, penalty: str = 'l1', bootstrap: int = 0,
                             proportion: float = 0.8, refresh=False) -> None:
    selected_path = "temp/arme_selected.pkl"
    df = pd.read_csv('temp/armenian_posts.csv', encoding='utf-8')
    df.dropna(0, 'any', subset=['Title', 'JobDescription', 'JobRequirment', 'RequiredQual'], inplace=True)
    df['description'] = df['JobDescription'].astype('str') + " " + \
                        df['JobRequirment'].astype('str') + " " + \
                        df['RequiredQual'].astype('str')
    df.rename(columns={'jobpost': 'ID', 'Title': 'title'}, inplace=True)
    df = df.filter(['ID', 'title', 'description'])
    process_filter_dump(df, inv_reg, penalty, bootstrap, proportion, refresh, selected_path, name='arme')


def main():
    from static.python.validation import cluster_purity
    import numpy as np
    # nyc_process_filter_dump(penalty='l1', bootstrap=0, proportion=0.9, inv_reg=3)
    cluster_dump('KMeans', *NYC_CLUSTER_PATHS, k=60, dynamic=False, min_size=10,
                 max_size=50, max_cycles=10)
    with open('temp/nyc_clustered.pkl', 'rb') as f:
        cluster_data = dill.load(f)
    print(f'purity={cluster_purity(cluster_data.df, cluster="KMeans")}')
    logging.info(cluster_purity(cluster_data.df, cluster="KMeans"))
    print(np.var(cluster_data.df.groupby('KMeans').nunique()))
    logging.info(np.var(cluster_data.df.groupby('KMeans').nunique()))


if __name__ == '__main__':
    main()
