# Credit_Risk_Analysis
supervised machine learning w/ python, jupyter notebook

## Purpose:
The purpose of this project was to use python and jupyter notebook to gain a basic understanding of supervised machine learning by building different prediction models and then evaluating them in their use to predict credit card risk.  This included:
* learning how a machine learning algorithm is used in data analytics
* Creating training and test groups from a given data set
* Implementing linear regression, logistic regression, decision tree, random forest, oversampling, undersampling, and support vector machine algorithms
* Interpreting the results of the models by using performance indicators such as: Accuracy scores, Precision, recall, F1-scores, and confusion matrices.
* Comparing the advantages and disadvantages of each supervised learning algorithm
* Determining which supervised learning algorithm is best used for a given data set or scenario
* Using ensemble and resampling techniques to improve model performance

## Analysis:

### Overview of the analysis: Explain the purpose of this analysis.
In order to achieve the above, I analyzed credit card risk data for a peer-to-peer lending services company, called LendingClub.  The dataset was a csv file containing 115,676 rows of data, and 86 columns initially.  As part of the pre-processing, I used the get_dummies method to convert the string values of multiple columns into numerical values, resulting in 95 columns(features), once the target column (loan_status) was removed.  I then used the following models to predict low_risk or high_risk loan_status, based on the previously mentioned 95 features:

* Naive Random Sampling (RandomOverSampler from imblearn)
* Synthetic Minority Oversampling Technique (SMOTE from imblearn)
* Undersampling (ClusterCentroids from imblearn)
* Combination Over and undersampling (SMOTEENN from imblearn)
* Ensemble Learning (BalancedRandomForestClassifier from imblearn)
* Ensemble Learning (EasyEnsembleClassifier from imblearn)

For each model: 
* I split the dataset into training and testing data, checked the balance of target values (value_count()), 
* generated X_train, X_test, y_train, y_test datasets with train_test_split, and printed the low_risk and high_risk values for the y_training data;
* trained the data to each model, and generated y_predictions
* calculated accuracy scores, confusion matrix, and classification_reports
* for the BalancedRandomForestClassifier model, I also listed the features with their corresponding importance values in descending order


### Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

Model 1: Naive Random Sampling (RandomOverSampler) <br>
Balanced Accuracy Score: 0.6614329112986135 <br>
Precision Score: high_risk - 0.01; low_risk - 1.0 <br>
Recall Score: high_risk - 0.72; low_risk - 0.60 <br>

![This is an image]()

Model 2: Synthetic Minority Oversampling Technique (SMOTE) <br>
Balanced Accuracy Score: 0.6581159869962674 <br>
Precision Score: high_risk - 0.01; low_risk - 1.0 <br>
Recall Score: high_risk - 0.62; low_risk - 0.69 <br>

![This is an image]()

Model 3: Undersampling (ClusterCentroids) <br>
Balanced Accuracy Score: 0.5442661782548694 <br>
Precision Score: high_risk - 0.01; low_risk - 1.0 <br>
Recall Score: high_risk - 0.69; low_risk - 0.40<br>

![This is an image]()

Model 4: Combination Over and undersampling (SMOTEENN) <br>
Balanced Accuracy Score: 0.6449163069955265 <br>
Precision Score: high_risk - 0.01; low_risk - 1.0 <br>
Recall Score: high_risk - 0.72; low_risk - 0.57 <br>

![This is an image]()

Model 5: Ensemble Learning (BalancedRandomForestClassifier) <br>
Balanced Accuracy Score: 0.7885466545953005 <br>
Precision Score: high_risk - 0.03; low_risk - 1.0 <br>
Recall Score: high_risk - 0.70; low_risk - 0.87 <br>

![This is an image]()

Model 6: Ensemble Learning (EasyEnsembleClassifier) <br>
Balanced Accuracy Score: 0.9316600714093861 <br>
Precision Score: high_risk - 0.09; low_risk - 1.0 <br>
Recall Score: high_risk - 0.92; low_risk - 0.94 <br>

![This is an image]()


### Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.
