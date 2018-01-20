# News Bias Classifier
Trains on "all-the-news" data from Kaggle; classifies news articles on the political scale of left or right.


<<<<<<< HEAD

Run the models:
To run any of the models, the raw Kaggle news article data in csv form must be in the relative path to the date EXAMPLE: ../cs221-data/read/data. make sure that your relative path to the data is correct!

To run a supervised model without cross_validation, choose a method: majority (majority), naive bayes (nb), domain-specific linear classification (ds), or word vectors linear classification (wv), then run the following command:
python supervised.py relative/data/path ./labeled_data.txt [method]

To run a supervised model with cross_validation, choose a method: majority (majority), naive bayes (nb), domain-specific linear classification (ds), or word vectors linear classification (wv), then run the following command:
python cross_validation.py relative/data/path ./labeled_data.txt [method]

To run a semisupervised model with naive bayes
run:
python semisupervised_nb.py relative/data/path ./labeled_data.txt 
add an optional cv flag for cross validation and error bars

To run a semisupervised model with linear classification with DSK
run:
python semisupervised_sgd.py relative/data/path ./labeled_data.txt 
add an optional cv flag for cross validation and error bars

To run the unsupervised naive bayes model, run the command:
python unsupervised.py relative/data/path ./labeled_data.txt
=======
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
>>>>>>> ac0d9c6421f0489c2562dd73c72930940771fc40
