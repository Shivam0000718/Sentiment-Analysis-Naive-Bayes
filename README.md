# Sentiment-Analysis-Naive-Bayes

# Overview
This repository contains a Machine Learning model for sentiment analysis using the Naive Bayes method. The model is trained on a dataset of 70,000 samples. Sentiment analysis is the process of determining the emotional tone behind a piece of text, whether it's positive, negative, or neutral.

# Libraries Used

1. pandas: For data manipulation and analysis.
2. numpy: For numerical computing.
3. seaborn: For data visualization.
4. matplotlib: For plotting graphs and charts.
5. Beautiful Soup (bs4): For scraping and parsing HTML.
6. re: For regular expressions.
7. nltk: For natural language processing tasks such as tokenization and stemming.
8. spacy: For advanced natural language processing tasks.
9. wordcloud: For generating word clouds.
10. sklearn: For machine learning algorithms and tools.
11. textblob: For processing textual data and sentiment analysis.

# Preprocessing Steps 

1. Data Cleaning: HTML tags are removed using BeautifulSoup. Text data is cleaned by removing special characters and symbols.
2. Tokenization: The text is tokenized into words and sentences using NLTK's word_tokenize and sent_tokenize functions.
3. Normalization: Words are converted to lowercase and punctuation is removed.
4. Stopword Removal: Common stopwords are removed using NLTK's stopwords corpus.
5. Stemming: Words are reduced to their root form using NLTK's PorterStemmer.
6. Vectorization: Text data is converted into numerical features using TF-IDF vectorization.

# Model Evaluation

The model's performance is evaluated using accuracy metrics. However, it has been observed that the current model considers neutral sentiment as negative due to the quality of the dataset. Therefore, there's room for improvement in the accuracy by adjusting the threshold for classifying neutral sentiment.

# Future Improvements 

To improve the accuracy of the sentiment analysis model, the following steps can be taken:

1. Fine-tuning Threshold: Adjust the threshold for classifying neutral sentiment to improve the accuracy.

2. Data Enhancement: Acquire a higher quality dataset with a more balanced distribution of sentiments.

3. Model Tuning: Experiment with different machine learning algorithms and hyperparameters to find the optimal configuration for sentiment analysis.

4. Feature Engineering: Explore additional features or linguistic cues that may improve sentiment classification accuracy.

# Usage 

1. Clone this repository to your local machine.
2. Install the required dependencies listed in the requirements.txt file.
3. Run the sentiment_analysis.py script to train and evaluate the sentiment analysis model.
4. Modify the model and preprocessing steps as needed to improve accuracy.
