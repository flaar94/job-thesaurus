from process_jobs import clean_title, my_lemmatizer, filter_postings
from frontend import find_from_posting
import pandas as pd
from joblib import load
import pytest

POST = """
    The Bureau of Water and Sewer Operation's Connections and Permitting Unit is seeking to hire five Civil
    Engineering Interns to serve as Inspectors. Under direct supervision, with little latitude for independent
    judgment, the selected candidates will perform engineering work of moderate difficulty and responsibility, but
    will not be limited to the following:   Conduct objective and thorough field inspections throughout the five
    boroughs to ensure methods of construction, and materials and workmanship used by the licensed plumbers or
    contractors fully conform to the current standards and specifications;  Maintain detailed and accurate records of
    inspections and reports, reports accurately on the events that transpire, secure and evaluate all facts and arrive
    at a sound conclusion  Develop detailed knowledge of all application and permit procedures related to water and
    sewer connections and installations.  Provide technical engineering support to other DEP units as necessary.
    Prepares oral and written reports as required.  Engages in studies, investigation or examinations related to the
    engineering functions or activities of the department, including the connection & Permitting."
    """


def test_clean_title():
    assert clean_title('Maintenance Worker - Technical Services-Heating Unit') == 'Maintenance Worker'
    assert clean_title('      Civil Engineer 5') == 'Civil Engineer'
    assert clean_title('Counsel II      ') == 'Counsel'
    assert clean_title('Project Manager, department of defense') == 'Project Manager'
    assert clean_title('.net       developer    ') == '.net Developer'
    assert clean_title('Senior Policy Advisor, Energy Finance and Affordability') == 'Policy Advisor'
    assert clean_title('Associate Public Health Sanitarian 2') == 'Public Health Sanitarian'
    assert clean_title('Senior Counsel   ') == 'Counsel'
    assert clean_title('Mid-Level Mobile Developer') == 'Mobile Developer'
    assert clean_title('Front-End Developer') == 'Front-end Developer'
    assert clean_title(
        "Forest Expert  for 'Integrated Biodiversity Management in the 'South Caucasus''") == 'Forest Expert'
    # assert clean_title('Programmer / Developer') == 'programmer'
    # assert clean_title('project website developer/ designer') == 'project website developer'
    assert clean_title('Net C#/ C++ Senior Software Developer') == 'Net C#/c++ Software Developer'
    assert clean_title('Entertainment & Sport Manager') == 'Entertainment and Sport Manager'
    # assert clean_title('Accounting / Finance Intern') == 'accounting/finance intern'
    # assert clean_title('Babysitter/ Governess') == 'babysitter/governess'
    assert clean_title('C++ Senior Software Developer') == 'C++ Software Developer'
    assert clean_title('Cashier in Yerevan') == 'Cashier'
    assert clean_title('Staff Writer: News & Politics') == 'Staff Writer'
    assert clean_title('clerical associate 3') == 'Clerical Associate'


def test_lemmatize():
    assert my_lemmatizer('He says you did it') == 'he say you do it'
    assert my_lemmatizer('I am me, not you') == 'i be me not you'
    assert my_lemmatizer(
        'The directors directed the director to direct') == 'the director direct the director to direct'


def test_processing():
    df = pd.DataFrame({'ID': [1, 2, 3, 4, 5],
                       'title': ['Civil Engineer 1', 'Manager, HR department', 'Civil Engineer', 'Senior Counsel   ',
                                 'Author'],
                       'description': ['This is a test', 'This is a test', 'Be careful',
                                       'But don\'t, don\'t be TOO careful',
                                       'Life is ephemeral']})
    df1 = filter_postings(df, min_postings=2)
    assert list(df1.title) == ['Civil Engineer', 'Civil Engineer']
    assert list(df1.description) == ['This is a test', 'Be careful']
    df2 = filter_postings(df, min_postings=1)
    assert list(df2.title) == ['Civil Engineer', 'Manager', 'Civil Engineer', 'Counsel', 'Author']
    assert list(df2.description) == ['This is a test', 'This is a test', 'Be careful',
                                     'But don\'t, don\'t be TOO careful',
                                     'Life is ephemeral']


# def test_job_processing():
#     df = pd.DataFrame({'ID': [1, 2, 3, 4, 5],
#                        'title': ['Civil Engineer 1', 'Manager, HR department', 'Civil Engineer', 'Senior Counsel   ',
#                                  'Author'],
#                        'description': ['This is a test', 'This is a test', 'Be careful',
#                                        'But don\'t, don\'t be TOO careful',
#                                        'Life is ephemeral']})
#     x, y, vectorizer = process_jobs(df, 1, 1, 1)


def test_find_from_posting():
    selected, vectorizer, mask, model, df = load("temp/arme_clustered.pkl", 'rb')
    assert find_from_posting(POST, vectorizer, mask, model, df)
