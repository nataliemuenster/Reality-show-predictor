# News Bias Classifier
Trains on "all-the-news" data from Kaggle; classifies news articles on the political scale of left or right.

#Run the models:
To run any of the models, the raw Kaggle news article data in csv form must be in the relative path: ../cs221-data/read/data.

To run a supervised model, choose a method: majority (majority), naive bayes (nb), linear classification (lin), or word vectors linear classification (wv), choose whether to use cross validation (y/n), then run the following command: #SGD --> LIN
python supervised.py ../cs221-data/read-data/ ./labeled_data.txt [method] [y/n]

To run a semisupervised model, choose a method: naive bayes (nb), linear classification (lin), or word vectors linear classification (wv) (DO WE HAVE WV HERE??), whether to use cross validation (y/n), then run the following command:
_________

To run the unsupervised naive bayes model, run the command:
python unsupervised.py ../cs221-data/read-data/ ./labeled_data.txt


TODO FORMAT RESTRUCTURE:
-rename baseline.py to supervised.py
-move cross validation so its a flag in supervised
-rename sgd to lin
-resample small amount of data for TA grading and running?