from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField, SelectField
from wtforms.widgets import TextArea
from wtforms.validators import DataRequired, Length, NumberRange
from static.python.default_postings import civil_engineering_intern


class TickerForm(FlaskForm):
    ticker = StringField("Ticker", validators=[DataRequired(), Length(1, 5)])
    month = IntegerField("Month", validators=[DataRequired(), NumberRange(1, 12)])
    year = IntegerField("Year", validators=[DataRequired(), NumberRange(1972, 2018)])
    submit = SubmitField("Submit")


class JobPostForm(FlaskForm):
    job_post = StringField("Job Post",
                           default=civil_engineering_intern.text,
                           validators=[DataRequired("Please enter your name"),
                                       Length(50, message=
                                       "The length of the job posting should be at least 50 characters")],
                           widget=TextArea())

    options = SelectField("Options", choices=[('title', 'Title Suggestions'),
                                              ('highlight', 'Highlight'), ('combined', 'Combined View')])
    data_source = SelectField("Data Source", choices=[('nyc', 'Official NYC'), ('armenian', 'Armenian')])
    submit = SubmitField("Submit")


class TitleForm(FlaskForm):
    title = StringField("Job Title", default='Civil Engineering Intern', validators=[DataRequired(), Length(1, 50)])
    data_source = SelectField("Data Source", choices=[('nyc', 'Official NYC'), ('armenian', 'Armenian')])
    submit = SubmitField("Submit")


class MedoidForm(FlaskForm):
    data_source = SelectField("Data Source", choices=[('nyc', 'Official NYC'), ('armenian', 'Armenian')])
    submit = SubmitField("Submit")


class HighlightForm(FlaskForm):
    highlight = SubmitField("Highlight!")


class ShowAll(FlaskForm):
    submit = SubmitField("Show All?")
