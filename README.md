# Spam-Email-Detection-using-Machine-Learning
A supervised classification pipeline to classify emails as spam or non-spam on the Enron email dataset. 

# Preprocessing
The first stage in the pipeline is to preprocess and clean the dataset.
The data is split into training (70%) and test (30%) sets, duplicate and empty emails are handled appropriately. Appropriate measures are taken to ensure that the test set is not biased in any way and some statistics are recorded on the resulting training and test sets.

# Feature extraction
In here, some exploratory data analysis is performed. Insights are gained from the data into features that could be useful for classification.
The top-20 most frequently used words are obtained from training sets of both spam and non-spam emails. Some common occurence words are removed from the obtained list to look out for stronger keypoints. The email lengths distribution in spam and non-spam emails are compared using a boxplot. TF-IDF Vectorizer was used to extract the features in comparison to Countvectorizer.

# Supervised classification
A supervised classification model is trained on the features and validation accuracy is calculated using cross validation. The data is split into 20 folds for each model for cross validation scores. Several classifiers are used to compare their performance including Naive Bayes (MultinomialNB), Logistic Regression, Support Vector Machine(SVM). Different combinstions of hyperparameters and sets of features are optimized.

# Model evaluation
Different metrics were used for evaluation of performance of our classifier. AUC, ROC and the Normalized Confusion matrix are the three metrics which are considered.

# Results
Naïve Bays gives an accuracy of 98.741%. So, 424 emails were classified incorrectly.
Logistic Regression gives an accuracy of 98.66%. So, 451 emails were classified incorrectly.
SVM gives an accuracy of 99.11%. So, 300 emails were classified incorrectly.
