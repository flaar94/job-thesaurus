import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
import dill


def cluster_purity(df: pd.DataFrame, cluster: str = "KMeans", unique_id: str = "ID", title: str = "title") -> int:
    grouped = df[[title, cluster, unique_id]].groupby([title, cluster]).count() / \
              df[[title, cluster, unique_id]].groupby([title]).count()
    grouped.head()
    grouped = grouped.groupby(title).max()
    job_count = df[[title, unique_id]].groupby([title]).count()
    job_count = job_count.rename({unique_id: "Count"}, axis=1)
    grouped = grouped.merge(job_count, on=title)
    grouped = grouped.dropna(1, "all")
    purity = (grouped[unique_id] * grouped["Count"]).sum() / grouped["Count"].sum()
    return purity


# def variance(df: pd.DataFrame, cluster: str = 'KMeans'):


def view_kmeans_graph(selected: np.array, validation_type: str = 'purity', df=None, max_k: int = 300, step=10,
                      start=10) -> tuple:
    scores = []
    temp_df = df.copy() if df is not None else None
    if validation_type == 'moment':
        for i in range(start, max_k, step):
            scores.append(KMeans(n_clusters=i, init="k-means++", random_state=i).fit(selected).score(selected))
    elif validation_type == 'purity':
        for i in range(start, max_k, step):
            temp_df['KMeans'] = KMeans(n_clusters=i, init="k-means++", random_state=i).fit_predict(selected)
            scores.append(cluster_purity(temp_df))
            print(f'k={i}, score={scores[-1]}')
    plt.plot(range(start, max_k, step), scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.show()
    return list(range(start, max_k, step)), scores


def view_gmm_graph(selected: np.array, validation_type: str = 'default', df: pd.DataFrame = None,
                   n_init: int = 5, max_k=150) -> tuple:
    scores = []
    temp_df = df.copy() if df is not None else None
    temp_selected = selected.toarray()
    if validation_type == 'default':
        for i in range(10, max_k, 10):
            model = BayesianGaussianMixture(i, covariance_type='diag',
                                            n_init=n_init,
                                            init_params='kmeans',
                                            random_state=12)
            scores.append(model.fit(temp_selected).score(temp_selected))
            print(f'k={i}, score={scores[-1]}')
    elif validation_type == 'purity':
        for i in range(10, max_k, 10):
            model = BayesianGaussianMixture(i, covariance_type='diag',
                                            n_init=n_init,
                                            init_params='kmeans',
                                            random_state=12)
            temp_df['KMeans'] = model.fit_predict(temp_selected)
            scores.append(cluster_purity(temp_df))
            print(f'k={i}, score={scores[-1]}')
    plt.plot(range(10, max_k, 10), scores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.show()
    return list(range(10, max_k, 10)), scores


if __name__ == '__main__':
    from joblib import load
    from bokeh.layouts import gridplot
    from bokeh.plotting import figure, show, output_file
    from bokeh.models import Span

    # selected, vectorizer, mask, df = load('temp/nyc_selected.pkl')
    #
    # moment_data = view_kmeans_graph(selected, df=df, validation_type='moment', max_k=250, step=5, start=5)
    # purity_data = view_kmeans_graph(selected, df=df, validation_type='purity', max_k=250, step=5, start=5)
    # with open('temp/validation_scores.pkl', 'wb') as f:
    #     dill.dump((moment_data, purity_data), f)

    moment_data, purity_data = load('temp/validation_scores.pkl')
    output_file('_model_selection.html')
    p1 = figure(x_axis_type="linear", title="Second Moment vs. Number of Clusters")
    p1.grid.grid_line_alpha = 1
    p1.xaxis.axis_label = 'Number of Clusters'
    p1.yaxis.axis_label = 'Second Moment Score'

    p1.line(moment_data[0], moment_data[1], color='blue', legend='Moment')
    p1.legend.location = "top_left"
    selected_k = Span(location=60, dimension='height', line_color='red', line_width=1)
    p1.add_layout(selected_k)

    p2 = figure(x_axis_type="linear", title="Purity vs. Number of Clusters")
    p2.grid.grid_line_alpha = 0.7
    p2.xaxis.axis_label = 'Number of Clusters'
    p2.yaxis.axis_label = 'Purity Score'

    p2.circle(purity_data[0], purity_data[1], color='#B2DF8A', legend='Purity')
    p2.legend.location = "top_right"
    p2.add_layout(selected_k)

    par = np.polyfit(purity_data[0], purity_data[1], 2, full=True)
    slope = par[0][0]
    intercept = par[0][1]
    y_predicted = [par[0][0] * i ** 2 + par[0][1] * i + par[0][2] for i in purity_data[0]]
    p2.line(purity_data[0], y_predicted, color='blue', legend='Quadratic Regression Line')

    window_size = 30
    window = np.ones(window_size) / float(window_size)
    show(gridplot([[p1, p2]], plot_width=500, plot_height=400))
