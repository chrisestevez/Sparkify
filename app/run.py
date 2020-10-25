import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar



app = Flask(__name__)

# load data
df = pd.read_csv('app/data/features.csv',usecols=['Features','Importance'])


@app.route('/')
@app.route('/index')
def index():
    """Website index.
    Returns:
        html: Webpage with plot.
    """
    
    # extract data needed for visuals
    
    importance_value = df['Importance']
    
    feature_value = list(df['Features'])
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=feature_value,
                    y=importance_value
                )
            ],

            'layout': {
                'title': 'Feature importance for GBTClassifier',
                'yaxis': {
                    'title': "Importance"
                },
                'xaxis': {
                    'title': "Features"
                }
            }
        }
    ]
    

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



def main():
    """Executes Flask app.
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()