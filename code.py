# Import necessary libraries
import os
import numpy
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib.pyplot import plot, figure, xlabel, ylabel, title, boxplot, show, bar
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import isfile, join
from collections import Counter
import pickle
import scikitplot as skplt

#Load the enron dataset
root_dir = "C:/Users/Guest/data/"
#Load all files
all_files = [f for f in os.listdir(root_dir) if isfile(join(root_dir, f))]
#Lenth of the dataset
all_mail_length = len(all_files)
print("Total number of emails: %d " %all_mail_length)
spam_number=0
ham_number=0
label=[]
# Number of Spam and ham in the dataset
for number in range (all_mail_length):
    if "spam" in all_files[number]:
        #Our labels
        label.append(1)
        spam_number=spam_number+1
    else:
        # Our labels
        label.append(0)
        ham_number=ham_number+1
# Print out the readings
print("Number of SPAM emails: %d" %spam_number)
print("Number of HAM emails: %d" %ham_number)

# Split the data into train and test
train_split, test_split, train_labels, test_labels = train_test_split(all_files, label, train_size = 0.7, test_size = 0.3, shuffle = True)

# Save the model
for saving in range(len(train_split)):
    save= pickle.dumps(saving)
for saving in range(len(test_split)):
    save= pickle.dumps(saving)
spam_train=0
ham_train=0
spam_test=0
ham_test=0

# Number of ham and spam in training set
for length in range(len(train_split)):
    # For ever spam file
    if "spam" in train_split[length]:
        spam_train=spam_train+1
    else:
        #For every ham file
        ham_train=ham_train+1
# Number of ham and spam in testing set
for length in range(len(test_split)):
    if "spam" in test_split[length]:
        spam_test=spam_test+1
    else:
        ham_test=ham_test+1
# Print out the values
print("Number of SPAM emails in training set: %d" %spam_train)
print("Number of HAM emails in training set: %d" %ham_train)
print("Number of SPAM emails in testing set: %d" %spam_test)
print("Number of HAM emails in testing set: %d" %ham_test)

# Top 20 word occurence in spam and ham in training set
common_ham_words=[]
common_spam_words=[]
# Top 20 Spam words
for top in range (len(train_split)):
    if "spam" in train_split[top]:
        #For opening every file
        path = os.path.join(root_dir, train_split[top])
        with open(path, encoding = 'latin-1') as f:
            #Read from the file
            spam_data = f.read()
            #This part was tried but was computationally expensive
            # common_spam_words += nltk.word_tokenize(spam_data)
            # dictionary=[w.lower() for w in common_spam_words if w.isalpha()]
            # dictionary=[w for w in dictionary if w>3]
        # dictionary=Counter(dictionary)
    # dictionary=dictionary.most_common(20)
        #Split elements in file
        common_spam_words += spam_data.split(" ")
#Replace and then remove non-alphabetic and words less than 3 letter
for top in range (len(common_spam_words)):
    if not common_spam_words[top].isalpha():
        common_spam_words[top] =""
    if len(common_spam_words[top]) < 3:
        common_spam_words[top] = ""
# Dictionary to keep count
dictionary_spam =Counter(common_spam_words)
del dictionary_spam[""]
# Top 20 words in dictionary
dictionary_spam =dictionary_spam.most_common(20)
top_spam_words =[]
top_spam_count =[]
#Separate the word and count of occurrence for plotting
for top in range (len(dictionary_spam)):
    [word, count]= dictionary_spam[top]
    #op spam words
    top_spam_words.append(word)
    #Count for them
    top_spam_count.append(count)
# Bar plot,give title,x-axis, y-axis name
figure()
bar(top_spam_words, top_spam_count, align='center', alpha=0.5, color=(1.0, 0.0, 0.0, 1.0))
title('Top 20 word occurrences of spam in training set')
ylabel('Number of appearances')
xlabel('Words')
show()

# Top 20 Ham words
for top in range (len(train_split)):
    if "ham" in train_split[top]:
        path = os.path.join(root_dir, train_split[top])
        with open(path, encoding = 'latin-1') as f:
            ham_data = f.read()
            #This part was tried but was computationally expensive
            # common_spam_words += nltk.word_tokenize(spam_data)
            # dictionary_ham=[w.lower() for w in common_spam_words if w.isalpha() ]
            # dictionary_ham=Counter(dictionary_ham)
            # dictionary_ham=dictionary_ham.most_common(20)
        common_ham_words += ham_data.split(" ")
        # Replace and then remove non-alphabetic and words less than 3 letter
for top in range (len(common_ham_words)):
    if not common_ham_words[top].isalpha():
        common_ham_words[top] =""
    if len(common_ham_words[top]) < 3:
        common_ham_words[top] = ""
# Dictionary to keep count
dictionary_ham=Counter(common_ham_words)
del dictionary_ham[""]
# Top20 words in Dictionary
dictionary_ham =dictionary_ham.most_common(20)
top_ham_words =[]
top_ham_count =[]
# Separate the word and count of occurrence for plotting
for top in range (len(dictionary_ham)):
    [word, count]= dictionary_ham[top]
    #Top Ham words
    top_ham_words.append(word)
    # Count for them
    top_ham_count.append(count)
# Bar plot,give title,x-axis, y-axis name
figure()
bar(top_ham_words, top_ham_count, align='center', alpha=0.5, color=(0.0, 1.0, 0.0, 1.0))
title('Top 20 word occurrences of Ham in training set')
ylabel('Number of appearances')
xlabel('Words')
show()

# BOX PLOT
length_spam=0
distance_spam=[]
length_ham=0
distance_ham=[]
# Extract length of files
for plotting in range (len(train_split)):
    if "spam" in train_split[plotting]:
        #For every spam file
        path = os.path.join(root_dir,train_split[plotting])
        # Open the file
        with open(path, encoding = 'latin-1') as f:
            #Read from file
            data=f.read()
            #Split the contents of file
            data=data.split()
            length_spam=len(data)
            #Add to the array
            distance_spam.append(length_spam)
    else:
        # For every ham file
        path = os.path.join(root_dir, train_split[plotting])
        with open(path, encoding='latin-1') as f:
            data = f.read()
            data = data.split()
            length_ham = len(data)
            distance_ham.append(length_ham)
# Plotting for spam
figure()
boxplot(distance_spam)
show()
# Plotting for ham
boxplot(distance_ham)
show()


train_features=[]
#Iterate to get all features
for feature in range (len(train_split)):
    path = os.path.join(root_dir, train_split[feature])
    with open(path, encoding = 'latin-1') as f:
        data=f.read()
        train_features.append(data)
length_features= len(train_features)
#  Countvectorizer
#tfid=CountVectorizer(stop_words='english', max_df=1.0, min_df=2, max_features=length_features)
# Initialise the Vectorizer
tfid = TfidfVectorizer(encoding= "latin-1",stop_words= "english", analyzer="word", max_features=length_features, vocabulary=None, max_df=1.0, min_df=2, dtype= numpy.int64, norm="l2")
# Fitting and transforming
train_vect=tfid.fit_transform(train_features)
# Array of training labels
train_label=numpy.array(train_labels)

test_features=[]
for feature in range (len(test_split)):
    path = os.path.join(root_dir, test_split[feature])
    with open(path, encoding='latin-1') as f:
        data = f.read()
        test_features.append(data)
# Transforming into vectors
test_vect=tfid.transform(test_features)
# Array of test labels
test_label=numpy.array(test_labels)



# Cross-validation
#Naive Bayes Cross Validation
classifier_naive= sklearn.naive_bayes.MultinomialNB().fit(train_vect,train_label)
# Score of cross-validation
cross_naive=cross_val_score(classifier_naive, train_vect, train_label, scoring='balanced_accuracy', cv=20)
print("Cross validation scores for Naive Bayes(MultinomialNB): ", cross_naive)
summing_nb=sum(cross_naive)
average_nb=summing_nb/20.0
print("Cross validation accuracy for Naive Bayes(MultinomialNB): %f" %average_nb)


# Logistic regression cross validation
classifier_lr= sklearn.linear_model.LogisticRegression().fit(train_vect,train_label)
# Score of cross-validation
cross_lr=cross_val_score(classifier_lr, train_vect, train_label, scoring='balanced_accuracy', cv=20)
print("Cross validation scores for Logistic regression: ", cross_lr)
summing_lr=sum(cross_lr)
average_lr=summing_lr/20.0
print("Cross validation accuracy for Logistic regression: %f" %average_lr)


# SVM cross validation
classifier_svm=sklearn.svm.SVC(kernel='linear').fit(train_vect,train_label)
# Score of cross-validation
cross_svm=cross_val_score(classifier_svm, train_vect, train_label, scoring='balanced_accuracy', cv=20)
print("Cross validation scores for Support Vector Machine: ", cross_svm)
summing_svm=sum(cross_svm)
average_svm=summing_svm/20.0
print("Cross validation accuracy for Support Vector Machine: %f" %average_svm)
# Saving the model
save=pickle.dumps(classifier_lr)
# Ploting scores with folds
figure()
plot(cross_naive, color="red")  # Red for Naive Bayes
plot((cross_lr), color="green")  # Green for logistic regression
plot((cross_svm), color="blue")  # Blue for SVM
xlabel("Number of folds")
ylabel("Cross Validation score")
show()

# ROC for test
#cl_svm=classifier_lr.probability = True

prediction=classifier_lr.predict_proba(test_vect)[:, 1]
t,r,threshold=sklearn.metrics.roc_curve(test_label, prediction)
# Plot ROC
figure()
plot(t, r)
plot([0,1], [0,1], 'k--')
xlabel("False positive rate")
ylabel("True positive rate")
title("ROC curve")
show()

# Calculate and print the AUC
auc=sklearn.metrics.auc(t,r)
print("AUC: ", auc)
# Confusion matrix
prediction_svm=classifier_lr.predict(test_vect)
# Plot the Normalized matrix
skplt.metrics.plot_confusion_matrix(test_label,prediction_svm, normalize=True)
show()



