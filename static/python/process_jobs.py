import pandas as pd
import numpy as np
import re
import string
from spacy.lang.en import English
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.feature_selection import SelectFromModel
from joblib import load
from collections import namedtuple, Counter
from sklearn.utils import resample
import dill
import os

ClusterData = namedtuple('ClusterData', ['selected', 'vectorizer', 'mask', 'model', 'df'])
L1RATIO = 0.7


def clean_title(title: str) -> str:
    """

    :param title: A job title
    :return: A job titles where trailing numbers, extra spaces, and anything following a ',' or '-' or '(' are removed
    """
    ignored_titles = ('associate', 'senior', 'junior', 'mid-level')
    replacements = {'&': 'and'}
    title = re.sub(' +', ' ', title).strip()
    title = title.upper().lower()
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"\s*/\s*", "/", title)
    title = re.split(r"(,|:|\s-\s|\sfor\b|\si+$|\d+$)", title)[0]
    title_list = title.split()
    if 'in' in title_list and title_list[0] != 'expert':
        title_list = title_list[:title_list.index('in')]
    title_list = [x for i, x in enumerate(title_list) if (x not in ignored_titles or i == len(title_list) - 1)]
    title_list = [x.capitalize() if x not in replacements else replacements[x] for x in title_list]
    title = ' '.join(title_list)
    return title.strip()


def clean_encoding(text: str) -> str:
    """

    :param text: text to be cleaned of encoding problems
    :return: A cleaned piece of text
    """
    text = text.replace(r'â€™', "'")
    text = text.replace('â€¢\t', '')
    return text


def my_lemmatizer(strng: str) -> str:
    """

    :param strng: A pieces of English text
    :return: The English text except all words have been lemmatized
    """
    # We convert a string into a bag-of-words style counter vector
    nlp = English()
    lemmas = [word.lemma_.upper().lower() for word in nlp(strng) if word.lemma_ not in string.punctuation]
    return ' '.join(lemmas)


def filter_postings(input_df: pd.DataFrame, min_postings: int = 6):
    df = input_df.copy()
    df['raw_title'] = df['title']
    df['title'] = df['title'].apply(clean_title)
    uniques = df[["ID", 'title']].groupby('title').count()
    uniques = uniques[uniques["ID"] >= min_postings]

    # Filter the dataframe to only include titles with sufficiently many job postings
    df = df[df["title"].apply(lambda x: x in uniques.index)]
    return df


def process_jobs(input_df: pd.DataFrame, min_postings: int = 6, min_num_docs: int = 5,
                 max_doc_freq: float = 0.8, name='nyc', refresh=False) -> tuple:
    """

    :param input_df: A dataframe with columns "title", "description", and 'ID'
    :param min_postings: All tiles with fewer than min_postings job postings will be filtered out
    :param min_num_docs: All words that appear in fewer than min_doc_freq documents will be filtered out
    :param max_doc_freq: ALL words that appear in more than max_doc_ratio of the documents will be filtered out
    :param name: name of the dataset
    :param refresh: whether to refresh the cache
    :return:
    """
    dump_path = f'temp/cache/process_{name}_{min_postings}_{min_num_docs}_{int(max_doc_freq * 100)}'
    if not refresh and os.path.exists(dump_path):
        return load(dump_path)
    else:
        input_df = input_df.sample(frac=1, random_state=52)
        df = filter_postings(input_df, min_postings)
        df['raw_description'] = df['description']
        df['description'] = df['description'].apply(my_lemmatizer)
        # Create a TFIDF vector from each description after lemmatizing
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', min_df=min_num_docs,
                                     max_df=max_doc_freq, sublinear_tf=True)
        vectorizer = vectorizer.fit(df['description'])
        x = vectorizer.transform(df["description"])
        y = df["title"]
        with open(dump_path, 'wb') as f:
            dill.dump((x, y, vectorizer, df), f)
    return x, y, vectorizer, df


def combine_masks(masks: list, proportion: float):
    """

    :param masks: A list of boolean numpy arrays representing tests
    :param proportion: The minimum proportion of masks a feature needs to be in before using it
    :return comb_mask: A mask obtained by picking all features that are in sufficient proportion in component masks
    """
    int_masks = [mask.astype('float') for mask in masks]
    prop_mask = sum(int_masks) / len(masks)
    comb_mask = np.array([True if feature >= proportion else False for feature in prop_mask])
    return comb_mask


def feature_selection(tfidf: np.array, titles: np.array, inv_reg: float = 3, random_state: int = 1, name='nyc',
                      penalty: str = 'l1', bootstrap: int = 0, proportion=0.8, refresh=False) -> tuple:
    """

    :param tfidf: A numpy array containing 'TFIDF' info
    :param titles: A list of titles
    :param inv_reg: the inverse regularization coefficient
    :param random_state: A random seed that starts for the logistic regression
    :param name: the name of the dataset (used to name the pickle file)
    :param penalty: The type of penalty to do features selection eg: 'l1' or 'elasticnet'
    :param bootstrap: The number of times to bootstrap, or 0 to not do bootstrapping
    :param refresh: choose whether to refresh the cache.
    :param proportion: Proportion of models the feature must appear in before putting it into our model
    :return: A list of words to ignore
    """
    l1_ratio = L1RATIO if penalty == 'elasticnet' else None
    if not bootstrap:
        logreg = LogisticRegression(random_state=random_state, penalty=penalty, C=inv_reg, solver='saga',
                                    multi_class="multinomial", n_jobs=4, max_iter=3000, l1_ratio=l1_ratio)
        logreg.fit(tfidf, titles)
        print(f'The accuracy of the model was {logreg.score(tfidf, titles)}')
        model = SelectFromModel(logreg, prefit=True)
        mask = model._get_support_mask()
    else:
        dump_path = f"temp/cache/bootstrap_{name}_{penalty}_{inv_reg if isinstance(inv_reg, int) else 'r' + str(int(100 / inv_reg))}_{bootstrap}"
        if not refresh and os.path.exists(dump_path):
            masks = load(dump_path)
        else:
            masks = []
            temp_logreg = LogisticRegression(penalty=penalty, C=inv_reg, solver='saga', multi_class="multinomial",
                                             n_jobs=4, max_iter=700, l1_ratio=l1_ratio)
            for i in range(bootstrap):
                temp_tfidf, temp_titles = resample(tfidf, titles, random_state=i)
                temp_logreg.random_state = random_state + i
                temp_logreg.fit(temp_tfidf, temp_titles)
                print(f'The accuracy of model {i + 1} was {temp_logreg.score(tfidf, titles)}')
                model = SelectFromModel(temp_logreg, prefit=True)
                masks.append(model._get_support_mask())
            with open(dump_path, 'wb') as f:
                dill.dump(masks, f)
        mask = combine_masks(masks, proportion)
    print(f'Number of features has been reduced from {tfidf.shape[1]} to {tfidf[:, mask].shape[1]}')
    filtered_tfidf = tfidf[:, mask]
    debiased_log = LogisticRegression(random_state=random_state, penalty=penalty, C=inv_reg, solver='saga',
                                      multi_class="multinomial", n_jobs=4, max_iter=700, l1_ratio=l1_ratio)
    debiased_log.fit(filtered_tfidf, titles)
    for j in range(filtered_tfidf.shape[1]):
        filtered_tfidf[:, j] *= np.linalg.norm(debiased_log.coef_[:, j])
    dense_tfidf = filtered_tfidf.toarray()
    scale = np.linalg.norm(dense_tfidf.T, axis=0).reshape([-1, 1])
    filtered_tfidf /= scale
    filtered_tfidf = sparse.csr_matrix(filtered_tfidf)
    return filtered_tfidf, mask


def test_clean_posting(text: str, vectorizer, mask):
    text = clean_encoding(text)
    text = my_lemmatizer(text)
    vect, = vectorizer.transform([text])
    vect = vect[0, mask]
    dense_vect = vect.toarray()
    scale = np.linalg.norm(dense_vect.T, axis=0).reshape([-1, 1])
    vect /= scale
    vect = sparse.csr_matrix(vect)
    return vect


class DynamicKMeans(KMeans):
    """
    Performs a dynamic version of Kmeans
    """

    def __init__(self, starting_n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto', min_size=0, max_size=float('inf'), max_cycles=5, jiggle=0.0001):
        super().__init__(starting_n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state,
                         copy_x, n_jobs, algorithm)
        self.min_size = min_size
        self.max_size = max_size
        self.max_cycles = max_cycles
        self.jiggle = jiggle

    def fit(self, X, y=None, sample_weight=None):
        # print('Huh?')
        super().fit(X, y, sample_weight)
        self.n_init = 1
        var = np.var(X.toarray())
        for i in range(self.max_cycles):
            print(self.score(X, y))
            counts = Counter(self.labels_)
            print(counts)
            # self.n_clusters
            changed = False
            for count in counts.values():
                if self.min_size <= count <= self.max_size:
                    continue
                elif count > self.max_size:
                    self.n_clusters += 1
                    changed = True
                elif count < self.min_size:
                    self.n_clusters -= 1
                    changed = True
            if not changed:
                break
            print(self.n_clusters)
            self.init = np.zeros([self.n_clusters, X.shape[1]])
            pointer = 0
            for x, count in counts.items():
                if self.min_size <= count <= self.max_size:
                    self.init[pointer, :] = self.cluster_centers_[x]
                    pointer += 1
                elif count > self.max_size:
                    self.init[pointer, :] = self.cluster_centers_[x] + np.random.randn(
                        1, X.shape[1]) * var * self.jiggle / self.n_clusters
                    self.init[pointer + 1, :] = self.cluster_centers_[x] + np.random.randn(
                        1, X.shape[1]) * var * self.jiggle / self.n_clusters
                    pointer += 2
                elif count < self.min_size:
                    continue
            super().fit(X, y, sample_weight)
        print(f'Ending clusters: {self.n_clusters}')
        return self


class DynamicGaussianMixture(GaussianMixture):
    """
    Performs a dynamic version of a spherical Gaussian Mixture
    """

    def __init__(self, n_components=1, tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10, min_size=0, max_size=float('inf'), max_cycles=5, jiggle=0.0001):
        super().__init__(n_components=n_components, covariance_type='spherical', tol=tol,
                         reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                         weights_init=weights_init, means_init=means_init, precisions_init=precisions_init,
                         random_state=random_state, warm_start=warm_start,
                         verbose=verbose, verbose_interval=verbose_interval)
        self.min_size = min_size
        self.max_size = max_size
        self.max_cycles = max_cycles
        self.jiggle = jiggle

    def fit(self, X, y=None, sample_weight=None):
        # print('Huh?')
        super().fit(X, y, sample_weight)
        self.n_init = 1
        var = np.var(X.toarray())
        for i in range(self.max_cycles):
            print(self.score(X, y))
            counts = Counter(self.labels_)
            print(counts)
            # self.n_clusters
            changed = False
            for count in counts.values():
                if self.min_size <= count <= self.max_size:
                    continue
                elif count > self.max_size:
                    self.n_clusters += 1
                    changed = True
                elif count < self.min_size:
                    self.n_clusters -= 1
                    changed = True
            if not changed:
                break
            print(self.n_clusters)
            self.init = np.zeros([self.n_clusters, X.shape[1]])
            pointer = 0
            for x, count in counts.items():
                if self.min_size <= count <= self.max_size:
                    self.init[pointer, :] = self.cluster_centers_[x]
                    pointer += 1
                elif count > self.max_size:
                    self.init[pointer, :] = self.cluster_centers_[x] + np.random.randn(
                        1, X.shape[1]) * var * self.jiggle / self.n_clusters
                    self.init[pointer + 1, :] = self.cluster_centers_[x] + np.random.randn(
                        1, X.shape[1]) * var * self.jiggle / self.n_clusters
                    pointer += 2
                elif count < self.min_size:
                    continue
            super().fit(X, y, sample_weight)
        print(f'Ending clusters: {self.n_clusters}')
        return self


def add_kmeans_labels(df: pd.DataFrame, selected: np.array, k: int = 60, dynamic=False, min_size=0,
                      max_size=float('inf'), max_cycles=5):
    if dynamic:
        means = DynamicKMeans(starting_n_clusters=k, random_state=12, min_size=min_size, max_size=max_size,
                              max_cycles=max_cycles)
    else:
        means = KMeans(n_clusters=k, random_state=12)
    labels = means.fit_predict(selected)
    df['KMeans'] = labels
    return means


def add_gmm_labels(df: pd.DataFrame, selected: np.array, k: int = 60, n_init=5):
    dense_selected = selected.toarray()
    model = BayesianGaussianMixture(k,
                                    covariance_type='diag',
                                    n_init=n_init,
                                    init_params='kmeans',
                                    random_state=12)
    labels = model.fit_predict(dense_selected)
    df['GMM'] = labels
    return model


def main():
    from joblib import load
    selected, vectorizer, mask, df = load('temp/arme_selected.pkl')
    means = DynamicKMeans(starting_n_clusters=60, min_size=5, max_size=300)
    means.fit(selected)


if __name__ == '__main__':
    main()


def zip_with_raw_titles(titles, df):
    raw_titles = []
    grouped = df[['title', 'raw_title']].groupby('title')
    for job in titles:
        raw_titles.append('\n'.join(sorted(list(grouped.get_group(job)['raw_title'].unique()))))
    return zip(titles, raw_titles)
