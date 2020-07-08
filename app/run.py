import json
import plotly
import pandas as pd
import plotly.graph_objs as go

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")

#Courtesy of Ioannis K Breier (https://github.com/jkarakas/Disaster-Response-Pipeline)

# create plotly figures 
def create_first_plot(df):
    """
    Create a plotly figure displaying the top 10 categories

    """

    top_categories = df.iloc[:, 4:].sum().sort_values(ascending=False)[:10]

    color_bar = 'DarkCyan'

    data = [go.Bar(
        x=top_categories.index,
        y=top_categories,
        marker=dict(color=color_bar),
        opacity=0.8
    )]

    layout = go.Layout(
        title="Top 10 categories",
        xaxis=dict(
            title='Categories',
            tickangle=45
        ),
        yaxis=dict(
            title='# of messages',
            tickfont=dict(
                color='DarkGreen'
            )
        )
    )

    return go.Figure(data=data, layout=layout), top_categories.index



def create_second_plot(df, top_10):
    """
    Create a stacked barchart displaying the number of messages per genre for the top 10 categories 
    """

    genres = df.groupby('genre').sum()[top_10]

    color_bar = 'DarkGreen'

    data = []
    for cat in genres.columns[1:]:
        data.append(go.Bar(
                    x=genres.index,
                    y=genres[cat],
                    name=cat)
                    )

    layout = go.Layout(
        title="No. of messages per genre for the top 10 categories",
        xaxis=dict(
            title='Genres',
            tickangle=45
        ),
        yaxis=dict(
            title='No. of messages per category',
            tickfont=dict(
                color=color_bar
            )
        ),
        barmode='stack'
    )

    return go.Figure(data=data, layout=layout)

graph1, top_10 = create_first_plot(df)
graph2 = create_second_plot(df, top_10)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
  
    graphs = [graph1, graph2]    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()