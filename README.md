# Disaster Repsonse NLP Pipeline 

### Motivation 
1. Perform data wrangling on Figure Eight datasets which contain real-world disaster messages.
2. Build NLP pipelines to create an optimized classifier and a web app with visualizations.
3. The result will be used by relevant disaster relief agencies to reduce response time and potentially save more lives.

### Screenshot of the Web App

### Project Structure

1. The ETL pipeline (./data directory)
- Combine the two datasets in CSV format
- Perform data wrangling and EDA
- Store the clean dataset in a SQLite database

Run ETL process with `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`

2. Machine Learning/NLP Pipeline (./models directory)
- Split the dataset into training and test sets
- Standardize, clean and tokenize the disaster messages
- Build a NLP pipeline
- Train the classifier and tune the hyperparameters with GridSearchCV

Run the NLP pipeline with `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

3. Show results of message classification and data visualizations with a Flask webapp (./app directory)
`run.py`

### Python pacakges used: 
- pandas
- plotly
- Flask
- sqlalchemy
- nltk: WordNetLemmatizer, stopwords, word_tokenize
- sklearn

