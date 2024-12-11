from flask import Flask, render_template, request
from main import predict_sentiment  # IMPORTING PREDICT_SENTIMENT FROM MAIN.PU

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':
        review = request.form['review']  # GET THE REVIEW FROM THE FORM
        if review:  # IF NOT EMPTY
            sentiment = predict_sentiment(review)  # CALLING FUNCTION TO PREDICT SENTIMENT
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
