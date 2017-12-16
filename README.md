# News Bias Classifier
Trains on "all-the-news" data from Kaggle; classifies news articles on the political scale of left or right.


Run the models:
To run any of the models, the raw Kaggle news article data in csv form must be in the relative path: ../cs221-data/read/data.

To run a supervised model without cross_validation, choose a method: majority (majority), naive bayes (nb), domain-specific linear classification (ds), or word vectors linear classification (wv), then run the following command:
python supervised.py ../cs221-data/read-data/ ./labeled_data.txt [method]

To run a supervised model with cross_validation, choose a method: majority (majority), naive bayes (nb), domain-specific linear classification (ds), or word vectors linear classification (wv), then run the following command:
python cross_validation.py ../cs221-data/read-data/ ./labeled_data.txt [method]

To run a semisupervised model, choose a method: naive bayes (nb), domain-specific linear classification (ds), then run the following command:
_________

To run the unsupervised naive bayes model, run the command:
python unsupervised.py ../cs221-data/read-data/ ./labeled_data.txt
