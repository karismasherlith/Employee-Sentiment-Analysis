# Employee-Sentiment-Analysis
This is a simple project that scrapes employee reviews from various companies, process and analyze them to build a machine learning model that predicts the sentiment of the review. The detailed description of each content in the repository is given below.

## Scraping.ipynb
This notebook consists of the code used for scraping reviews from CareerBliss.
- It uses libraries like requests, BeautifulSoup and pandas for web scraping and data manipulation.
- The scrape_reviews function extracts reviews from the website. This is done by identifying the unique class names that define each part of the review and then passing that class name inside the function to read it smoothly across multiple pages.
- The scraped reviews are then balanced to ensure that all sentiments have a equal distribution for model to learn properly.
- The balanced dataset is saved as 'balanced_reviews.csv'.

## Training and Testing.ipynb
This notebook cover the preparation of data, cleaning and preprocessing, model training, visualization and evaluation.
- The balanced dataset is loaded for cleaning. Each review is processed to lower it, removing unwanted spaces, symbols, emojis etc, removing stop words while excluding few words (important_words) that may have an impact on the review.
- The processed review is then again tokenized and lemmatized to convert the words to the simplest form.
- The ratings are then categorized into Sentiment groups (Positive, Negative and Neutral).
- The sentiment and rating distribution is visualised to see whether dataset has a normal distibution.
- Additional features like positive word count, negative word count, review length etc are extracted from the review with the help of libraries like opinion_lexicon.
- The tokenizer used is saved as a pickle file (tokenizer.pkl) and the reviews are padded and later split into training and testing datasets.
- Hyperparamter tuning is done with the help of Optuna to get the best hyperparameter values.
- A model with accuracy 92.06% is built and saved as sentiment_model.pkl
- The notebook also contains functions for preprocessing a new review and predicting its sentiment which is carried out to check the model accuracy for different test cases.
  
## main.py
This notebook contains the major functions required for cleaning, processing and predicting the sentiment of a new review.
- It first loads the saved model and tokenizer that was used during model training.
- preprocess_review() takes a review input and preprocesses it based on our training methods.
- predict_sentiment() takes the processed review and pads it and extracts the additional features that were used for prediction in training the model.
- The sentiment is then predicted using the saved model.
  
## index.html
This file consists of the html layout including the css. It contains a form which asks user to enter a review and 2 buttons for submitting and resetting. The submit button when clicked prints the sentiment of the entered review, while the reset button clears out any printed sentiment.

## app.py
This is the final Flask Application that uses the index.html as the template and predict_sentiment function from main.py to predict the sentiment of the entered review. The application can be run by using the command 'python app.py'

### Few Test Cases:
1. this is a great company i love working here! (MODEL RETURNS POSITIVE)
2. this is the WoRst Company ever, hate everything. sucks (MODEL RETURNS NEGATIVE)
3. good company but work environment is bad just ok (MODEL RETURNS NEUTRAL)
4. company is ok (MODEL RETURNS NEUTRAL)
