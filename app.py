from flask import Flask, render_template, request, flash, url_for
from static.python.forms import JobPostForm, TitleForm, HighlightForm
import os
import dill
from bs4 import BeautifulSoup

from static.python.frontend import find_from_posting_ordered, compute_medoids, highlight_text, find_from_title_ordered, \
    reorder_titles_vertical

app = Flask(__name__)
if os.path.exists('secret_key.py'):
    from secret_key import secret_key
else:
    secret_key = os.environ.get('SECRET_KEY', None)
app.config["SECRET_KEY"] = secret_key


@app.route('/')
def index():
    return render_template('home/index.html', about_url=url_for("details"), postings_url=url_for('postings'),
                           title='Home')


@app.route('/postings', methods=['GET', 'POST'])
def postings():
    """

    :return:
    """
    option = None
    data = None
    text = None
    form = JobPostForm()
    if request.method == 'POST':
        if form.validate():
            text = BeautifulSoup(form.job_post.data, features="html.parser").text
            option = form.options.data
            if form.data_source.data == 'armenian':
                data_url = u'static/pickle/arme_clustered.pkl'
                relevant_words_url = 'static/pickle/arme_relevant_words.pkl'
            elif form.data_source.data == 'nyc':
                data_url = u'static/pickle/nyc_clustered.pkl'
                relevant_words_url = 'static/pickle/nyc_relevant_words.pkl'
            else:
                raise Exception('data source not yet implemented')
            with open(data_url, 'rb') as f:
                cluster_data = dill.load(f)
            if option == 'title' or option == 'combined':
                data = find_from_posting_ordered(text, cluster_data)
                data = reorder_titles_vertical(data)
            if option == 'highlight' or option == 'combined':
                with open(relevant_words_url, 'rb') as f:
                    relevant_words = dill.load(f)
                text = highlight_text(text, relevant_words)
    return render_template('features/postings.html', form=form,
                           option=option, data=data, text=text,
                           title="Posting functions")


@app.route('/titles', methods=["GET", "POST"])
def titles():
    data = []
    form = TitleForm()
    if request.method == 'POST':
        if form.validate():
            text = form.title.data
            if form.data_source.data == 'armenian':
                data_url = u'static/pickle/arme_clustered.pkl'
            elif form.data_source.data == 'nyc':
                data_url = u'static/pickle/nyc_clustered.pkl'
            else:
                raise Exception('data source not yet implemented')
            cluster_data = dill.load(open(data_url, 'rb'))
            raw_data = find_from_title_ordered(text, cluster_data)
            if raw_data is None:
                flash("No matching title found in the data")
                raw_data = []
            raw_data = list(raw_data)
            if len(list(raw_data)) == 1:
                flash("This title is an outlier: no other titles are very similar")
            data = reorder_titles_vertical(raw_data)
    return render_template('features/titles.html', form=form, title="Title functions", data=data)


@app.route('/medoids', methods=["GET", "POST"])
def medoids():
    data = []
    form = TitleForm()
    if request.method == 'POST':
        if form.validate():
            if form.data_source.data == 'armenian':
                data_url = u'static/pickle/arme_clustered.pkl'
            elif form.data_source.data == 'nyc':
                data_url = u'static/pickle/nyc_clustered.pkl'
            else:
                raise Exception('data source not yet implemented')
            cluster_data = dill.load(open(data_url, 'rb'))
            _, *partial_data = cluster_data
            data = compute_medoids(cluster_data.df, cluster_data.selected)
    return render_template('features/medoids.html', form=form, title="Title functions", data=data)


@app.route('/details', methods=['GET', 'POST'])
def details():
    from static.python.default_postings import inspector, raw_inspector, civil_engineering_intern, Posting
    relevant_words_url = 'static/pickle/nyc_relevant_words.pkl'
    if request.method == 'GET':
        post1, post2 = inspector, civil_engineering_intern
    else:
        with open(relevant_words_url, 'rb') as f:
            relevant_words = dill.load(f)
        post1, post2 = raw_inspector, civil_engineering_intern
        post1 = Posting(post1.title, highlight_text(post1.text, relevant_words))
        post2 = Posting(post2.title, highlight_text(post2.text, relevant_words))
    return render_template('home/details.html', title='Details', form=HighlightForm(), post1=post1, post2=post2)


@app.route('/about')
def about():
    return render_template('home/about.html', title='About')


@app.route('/testing/<my_monkey>')
def monkey_business(my_monkey):
    return render_template('home/details.html', monkey=my_monkey)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
