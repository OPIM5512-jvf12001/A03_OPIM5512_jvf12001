import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import random
from collections import Counter
import seaborn as sns


df = pd.read_csv("california_housing_train.csv")
print(df.head())

# df['median_house_value'].hist()
# plt.show()

print("\nImbalanced Data Count")
df['median_house_value'] = np.where(df['median_house_value'] < 380000, 1, 0)
print(df['median_house_value'].value_counts())

# Split into X and y

X = df.drop('median_house_value', axis = 1)
y = df['median_house_value']

# Taking a second to look at the input feature data
print (X.describe())

# Sanity check to see that X and y are the correct size
print('\n', '*' * 90)
print('# of X rows: ',len(X))
print('# of y rows: ', len(y))

# Splitting the data into train and test as well as checking the resulting array shapes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 14)
print('\n', '*' * 90)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape) # looks good!

# Checking the distribution of y values in the training set, heavy skew towards 1s as expected. The ratio of 1 to 0 is very similar
# between training and test data
print('\n', '*' * 90)
print('y_train Distribution: ',Counter(y_train))
print('y_test Distribution: ',Counter(y_test))

# Using a float here to specify that min_samples_split should be a consistent percentage of the overall number of examples in the dataset
# This will allow me to comfortable use this same model when comparing the different sampling strategies as our total number of examples
# to train on will be much greater when oversampling than undersampling.
DTC_under = DecisionTreeClassifier(min_samples_split=0.01)
DTC_over = DecisionTreeClassifier(min_samples_split=0.01)
DTC_SMOTE = DecisionTreeClassifier(min_samples_split=0.01)

# Using Majority Undersampling on the training data using the imblearn RandomUnderSampler. A check shows that this was successful
undersample = RandomUnderSampler(sampling_strategy = 'majority')
X_under, y_under = undersample.fit_resample(X_train, y_train)
print('\n', '*' * 90)
print('y_under Distribution: ',Counter(y_under))

# Fitting the data and predicting
DTC_under.fit(X_under, y_under)
under_train_preds = DTC_under.predict(X_under)
under_test_preds = DTC_under.predict(X_test)

# Creating classification reports and a confusion matrix to evaluate how the model performed using the data that was balanced using
# Majority undersampling.
print('\n', '*' * 90)
print('Undersampling Train results: \n')
print(classification_report(y_under, under_train_preds))
print('Undersampling Test results: \n')
print(classification_report(y_test, under_test_preds))

print ('\n Majority Undersampling Confusion Matrix: ')
print(confusion_matrix(y_test, under_test_preds))
# *******************************************************************************************************************************
# Majority Undersampling Results Comments:
# Reviewing these results, it looks like this model learned the training data excellently but did terrible once it was run
# on the test dataset. The model is overfitting the training data. The confusion matrix shows almost no false positives 
# (predicted 1 but actually 0), but many false negatives (predicting 0 but actually 1). This is consistent with the 
# classification report showing that we have pretty good recall on detecting 0s but terrible precision as there
# are far too many predicted.

# *******************************************************************************************************************************


# Using Minority Oversampling on the training data using the imblearn RandomOverSampler. A check shows that this was successful
oversample = RandomOverSampler(sampling_strategy = 'minority')
X_over, y_over = oversample.fit_resample(X_train, y_train)
print('\n', '*' * 90)
print('y_over Distribution: ',Counter(y_over))

# Fitting the data and predicting
DTC_over.fit(X_over, y_over)
over_train_preds = DTC_over.predict(X_over)
over_test_preds = DTC_over.predict(X_test)

# Creating classification reports and a confusion matrix to evaluate how the model performed using the data that was balanced using
# Minority Oversampling.
print('\n', '*' * 90)
print('Oversampling Train results: \n')
print(classification_report(y_over, over_train_preds))
print('Oversampling Test results: \n')
print(classification_report(y_test, over_test_preds))

print ('\n Minority Oversampling Confusion Matrix: ')
print(confusion_matrix(y_test, over_test_preds))

# *******************************************************************************************************************************
# Minority Oversampling Results Comments:
# Overall, attempt 2 using Minority Oversampling is doing a slightly better job. The model is generalizing better as we can see
# the number of false negatives drop considerably, at the same time however we are getting more false positives.
# The f1 score of model 2 also slightly better than model 1. This is still likely overfitting however, as the model performs
# almost perfectly on the training data with an f1 score of a solid 1.00.

# *******************************************************************************************************************************


# Using SMOTE on the training data. A check shows that this was successful
over_smote = SMOTE(k_neighbors = 9)
X_smote, y_smote = over_smote.fit_resample(X_train, y_train)
print('\n', '*' * 90)
print('y_smote Distribution: ',Counter(y_smote))

# Fitting the data and predicting
DTC_SMOTE.fit(X_smote, y_smote)
smote_train_preds = DTC_SMOTE.predict(X_smote)
smote_test_preds = DTC_SMOTE.predict(X_test)

# Creating classification reports and a confusion matrix to evaluate how the model performed using the data that was balanced using
# SMOTE.
print('\n', '*' * 90)
print('SMOTE Train results: \n')
print(classification_report(y_smote, smote_train_preds))
print('SMOTE Test results: \n')
print(classification_report(y_test, smote_test_preds))

print ('\n SMOTE Confusion Matrix: ')
print(confusion_matrix(y_test, smote_test_preds))

# *******************************************************************************************************************************
# SMOTE Results Comments:
# Again performance is not great, but is similar to what we got when using basic Minority Oversampling
# We get a very similar f1 score on the test partition, while the model did not memorize the training partition quite as much. 
# We can also see that our false negatives and false positives held steady coming in very close to Minority Oversampling.
# To improve accuracy at this point would require tinkering with the model, or perhaps trying a fancier sampling technique

# *******************************************************************************************************************************
# Methodology Comparison Comments:
# Overall, I would say that performance was similar between all three methods on the test partition, with all three coming in at
# f1 scores around 0.51 or below. Majority Undersampling definitely did the worst in the case of f1 score and in the number
# of false negatives. In my opinion, basic Minority Oversampling takes the lead in this comparison edging out SMOTE
# ever so slightly.

# *******************************************************************************************************************************

# I am choosing to run SMOTE through the for loop, as I had thought it would perform the best before seeing the original comparison
# results. Perhaps I just chose an unlucky random state, but this will help provide some clarity on that.

# Initializing blank lists to hold the data from each run
precision_recs, recall_recs, f1_recs, accuracy_recs = [],[],[],[]

for i in range(1000):

    X_train_loop, X_test_loop, y_train_loop, y_test_loop = train_test_split(X, y, test_size = 0.2, random_state = i)

    # Utilizing SMOTE
    over_smote_loop = SMOTE(k_neighbors = 9)
    X_smote_loop, y_smote_loop = over_smote_loop.fit_resample(X_train_loop, y_train_loop)

    # Fitting the data and predicting
    DTC_SMOTE.fit(X_smote_loop, y_smote_loop)
    smote_preds_loop = DTC_SMOTE.predict(X_test)

    # Generating a classification report and extracting the data
    loop_report = classification_report(y_test, smote_preds_loop, output_dict = True)
    precision = loop_report['0']['precision']
    recall = loop_report['0']['recall']
    f1 = loop_report['0']['f1-score']
    accuracy = loop_report['accuracy']


    # Appending the extracted data to initialized lists
    precision_recs.append(precision)
    recall_recs.append(recall)
    f1_recs.append(f1)
    accuracy_recs.append(accuracy)


# Creating our plots to look at the bell curves.
sns.histplot(precision_recs, kde = True, color = 'blue', label = 'Precision Values')
sns.histplot(recall_recs, kde = True, color = 'orange', label = 'Recall Values')
sns.histplot(f1_recs, kde = True, color = 'green', label = 'f1 Values')
sns.histplot(accuracy_recs, kde = True, color = 'purple', label = 'Accuracy Values')
plt.title("Distribution of Model Performance on SMOTE Data")
plt.legend(loc='upper center')
plt.savefig('figs/smote_loop_performance.png', dpi=300, bbox_inches='tight')
plt.show()


# *******************************************************************************************************************************
# Final Comments:
# I am very pleased to see the results of the repeated experiment, I started with 30 runs and bumped it up until I no longer felt like waiting on the program.
# Through this, we can see in the generated histogram that random_state 42 which I chose for the original methodology comparison is actually a
# particularly poor performance for SMOTE. The f1 score I originally got is near the minimum of all f1 scores across the states that were tested in this loop.
# Addtionally, the precision and recall scores were also better in most other runs.

# *******************************************************************************************************************************