# Credit_Risk_Analysis
supervised machine learning w/ python, jupyter notebook

## Purpose:
The purpose of this project was to use python and jupyter notebook to gain a basic understanding of supervised machine learning by building different prediction models and then evaluating them in their use to predict credit card risk.  This included:
* learning how a machine learning algorithm is used in data analytics
* Creating training and test groups from a given data set
* Implementing the liniear regression, logistic regression, decision tree, random forest, and support vector machine algorithms
* Interpreting the results of the models by using performance indicators such as: Accuracy scores, Precision, recall, F1-scores, and confusion matrixes.
* Comparing the advantages and disadvantages of each supervised learning algorithm
* Determining which supervised learning algorithm is best used for a given data set or scenario
* Using ensemble and resampling techniques to improve model performance


### Overview of the analysis: Explain the purpose of this analysis.
In order to achieve the above, I analyzed credit card risk data for a peer-to-peer lending services company, called LendingClub.  The dataset was a csv file containing 115,676 rows of data, and 86 columns initially.  As part of the pre-processing, I used the get_dummies method to convert the string values of multiple columns into numerical values, resulting in 95 columns(features), once the target column (loan_status) was removed.  I then used the following models to predict low_risk or high_risk loan_status, based on the previously mentioned 95 features:

* Naive Random Sampling (RandomOverSampler from imblearn)
* Synthetic Minority Oversampling Technique (SMOTE from imblearn)
* Undersampling (ClusterCentroids from imblearn)
* Combination Over and undersampling (SMOTEENN from imblearn)
* Ensemble Learning (BalancedRandomForestClassifier from imblearn)
* Ensemble Learning (EasyEnsembleClassifier from imblearn)

For each model: 
I split the dataset into training and testing data, checked the balance of target values (value_count()), 
generated X_train, X_test, y_train, y_test datasets with train_test_split, and printed the low_risk and high_risk values for the y_training data;
trained the data to each model, and generated y_predictions
calculated accuracy scores, confusion matrix, and classification_reports
for the BalancedRandomForestClassifier model, I also listed the features with their corresponding importance values in descending order


### Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

### Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
