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

### Overview: 
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


### Results:

Model 1: Naive Random Sampling (RandomOverSampler) <br>
Balanced Accuracy Score: 0.6614329112986135 <br>
Precision Score: high_risk - 0.01; low_risk - 1.0 <br>
Recall Score: high_risk - 0.72; low_risk - 0.60 <br>

![This is an image](https://github.com/bartblack13/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/RandomOverSampler.png)
<br>
**Figure 1: RandomOverSampler Classification Report**<br><br>

Model 2: Synthetic Minority Oversampling Technique (SMOTE) <br>
Balanced Accuracy Score: 0.6581159869962674 <br>
Precision Score: high_risk - 0.01; low_risk - 1.0 <br>
Recall Score: high_risk - 0.62; low_risk - 0.69 <br>

![This is an image](https://github.com/bartblack13/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/SMOTE.png)
<br>
**Figure 2: SMOTE Classification Report**<br><br>

Model 3: Undersampling (ClusterCentroids) <br>
Balanced Accuracy Score: 0.5442661782548694 <br>
Precision Score: high_risk - 0.01; low_risk - 1.0 <br>
Recall Score: high_risk - 0.69; low_risk - 0.40<br>

![This is an image](https://github.com/bartblack13/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/ClusterCentroid.png)
<br>
**Figure 3: ClusterCentroids Classification Report**<br><br>

Model 4: Combination Over and undersampling (SMOTEENN) <br>
Balanced Accuracy Score: 0.6449163069955265 <br>
Precision Score: high_risk - 0.01; low_risk - 1.0 <br>
Recall Score: high_risk - 0.72; low_risk - 0.57 <br>

![This is an image](https://github.com/bartblack13/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/SMOTEENN.png)
<br>
**Figure 4: SMOTEENN Classification Report**<br><br>

Model 5: Ensemble Learning (BalancedRandomForestClassifier) <br>
Balanced Accuracy Score: 0.7885466545953005 <br>
Precision Score: high_risk - 0.03; low_risk - 1.0 <br>
Recall Score: high_risk - 0.70; low_risk - 0.87 <br>

![This is an image](https://github.com/bartblack13/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/BalancedRandomForest.png)
<br>
**Figure 5a: BalancedRandomForestClassifier Classification Report**<br><br>

![This is an image](https://github.com/bartblack13/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/importances%20BRFC.png)
<br>
**Figure 5b: BalancedRandomForestClassifier Feature Importances**<br><br>


Model 6: Ensemble Learning (EasyEnsembleClassifier) <br>
Balanced Accuracy Score: 0.9316600714093861 <br>
Precision Score: high_risk - 0.09; low_risk - 1.0 <br>
Recall Score: high_risk - 0.92; low_risk - 0.94 <br>

![This is an image](https://github.com/bartblack13/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/ADAboost.png)
<br>
**Figure 6: EasyEnsembleClassifier (ADAboost) Classification Report**<br><br>


### Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

To analyze the data anc predict whether or not an applicant was a high or low risk risk of credit card default, I employed 6 different models.  The dataset was extremely impbalanced with low risk applicants far outweighing high risk ones.  Prior to spliting into training and testing groups, I used the value_counts() function to check the balance of my target values, which yielded low_risk': 51366, 'high_risk': 246, where high_risk targets only amount to 0.5% of the total targets.  This low percentage was maintained after splitting into training and target groups.  The 6 models I used employed different over sampling, undersampling, or combination techniques.  For each model I generated accuracy values, precision scores, and recall/sensitivity scores.

Looking at the results above, we see that the accuracy scores ranged from 0.544-0.932, with EasyEnsembleClassifier method having the highest score.  But because of the imbalanced data, the accuracy score alone is not the best indication of a successful model.  All of the models produced a precision score for low_risk targets of 1.0 and a precision score ranging from 0.01-0.09 for high_risk targets, which is not surprising, again, because of the imbalance.  The EasyEnsembleClassifier had the highest high_risk precision score.  The recall/sensitivity score across all 6 models ranged from 0.40-0.94 for low_risk, and 0.62-0.92 for high_risk targets, with EasyEnsembleClassifier method having the highest scores for both targets.

This clearly indicates that the EasyEnsembleClassifier method was the strongest of the 6 models tested, however, all of the models proved to be weak learners in their precision capabilities. I would be cautious to recommend that LendingClub rely solely on the EasyEnsembleClassifier model for their prediction efforts.  While it did have a high accuracy score and recall score for both targets, it had very poor precision.  The company would have to weigh which score is more important.  Since precision measures how reliable a prediction is (probability that a target classified as high_risk is indeed high-risk), we see that the EasyEnsembleClassifier method only produced a 0.09 (9%) chance that a high_risk classification was true.  This means that a high number of low_risk applicants could be falsely classified as high_risk, thereby causing loss of substantial profits over time. 

