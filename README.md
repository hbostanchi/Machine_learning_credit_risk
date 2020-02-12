# Machine_learning_credit_risk
## Overview
In 2019, more than 19 million Americans had at least one unsecured personal loan. That’s a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.
In this module, we used Python to build and evaluate several machine learning models to predict credit risk. Being able to predict credit risk with machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.
## Resources
+	Data Source:
+	Software: Python and Scikit-learn
## Objectives
+	Explain how a machine learning algorithm is used in data analytics.
+	Create training and test groups from a given data set.
+	Implement the logistic regression, decision tree, random forest, and support vector machine algorithms.
+	Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms.
+	Compare the advantages and disadvantages of each supervised learning algorithm.
+	Determine which supervised learning algorithm is best used for a given data set or scenario.
+	Use ensemble and resampling techniques to improve model performance.
## Summary
Explain how a machine learning algorithm is used in data analytics.
Machine learning is the use of statistical algorithms to perform tasks such as learning from data patterns and making predictions. There are many different models.
+	A model is a mathematical representation of something that happens in the real world.
## Machine learning can be divided into three learning categories: 
supervised, unsupervised, and deep. For our purposes, we only discussed supervised and unsupervised learning.
+	Supervised learning deals with labeled data.
+	Unsupervised learning algorithms work with datasets without labeled outcomes.

### In supervised learning, the labels provide the correct answers. In unsupervised learning, such correct answers, or labels, aren’t provided.
### Supervised learning can be broadly divided into regression and classification.
+	Regression is used to predict continuous variables.The regression model’s algorithms attempt to learn patterns that exist among factors given. If presented with new data, the model will make a prediction, based on previously learned patterns from the dataset.
+	Classification is used to predict discrete outcomes.The classification model’s algorithms attempts to learn patterns from the data, and if the model is successful, gain the ability to make accurate predictions.
There is a major difference between regression and classification models. In regression a continuous variable can be any numerical value within a certain range. In classification, on the other hand, the target variable only has two possible values.
A basic pattern applies whether we’re using regression or classification:
+	A machine learning model is presented with a dataset.
+	The model algorithms analyze the data and attempt to identify patterns.
+	Based on these patterns, the model makes predictions on new data.
Create training and test groups from a given data set.
+Training dataset to learn from it.
+Testing dataset to assess its performance.
Implement the logistic regression, decision tree, random forest, and support vector machine algorithms.
Logistic regression predicts binary outcomes, meaning that there are only two possible outcomes. In other words, a logistic regression model analyzes the available data, and when presented with a new sample, mathematically determines its probability of belonging to a class. If the probability is above a certain cutoff point, the sample is assigned to that class. If the probability is less than the cutoff point, the sample is assigned to the other class.
Decision trees are used in decision analysis. They encode a series of true/false questions that are represented by a series of if/else statements and are one of the most interpretable models, as they provide a clear representation of how the model works.
Random Forests does not have a complex tree like the ones created by decision trees, a random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features.
Random forest algorithms are beneficial because they:
+	Are robust against overfitting as all of those weak learners are trained on different pieces of the data.
+	Can be used to rank the importance of input variables in a natural way.
+	Can handle thousands of input variables without variable deletion.
+	Are robust to outliers and nonlinear data.
+	Run efficiently on large datasets.
Support vector machine (SVM), like logistic regression, is a binary classifier. It can categorize samples into one of two categories. There is a strict cutoff line that divides one classification from the other.
Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms.
The results in the classification report:
+	Precision is the measure of how reliable a positive classification is. The precision can be determined by the ratio: TP/(TP + FP). A low precision is indicative of a large number of false positives.
+	Recall is the ability of the classifier to find all the positive samples. It can be determined by the ratio: TP/(TP + FN). A low recall is indicative of a large number of false negatives.
+	F1 score is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0 and the worst is 0.0
+	Support is the number of actual occurrences of the class in the specified dataset.
Compare the advantages and disadvantages of each supervised learning algorithm.
The Logistic regression model can make predictions beyond the range in the current data.
Decision trees are natural ways in which you can classify or label objects by asking a series of questions designed to zero in on the true answer. However, decision trees can become very complex and very deep, depending on how many questions have to be answered. Deep and complex trees tend to overfit to the data and do not generalize well.
Random Forests, simple trees, are weak learners because they are created by randomly sampling the data and creating a decision tree for only that small portion of data. And since they are trained on a small piece of the original data, they are only slightly better than a random guess. However, many slightly better than average small decision trees can be combined to create a strong learner, which has much better decision making power.
SVM works by separating the two classes in a dataset with the widest possible margins. The margins, however, are soft and can make exceptions for outliers. This stands in contrast to the logistic regression model. In logistic regression, any data point whose probability of belonging to one class exceeds the cutoff point belongs to that class; all other data points belong to the other class.
Determine which supervised learning algorithm is best used for a given data set or scenario.
Modeling is an iterative process: you may need more data, more cleaning, another model parameter, or a different model. It’s also important to have a goal that’s been agreed upon, so that you know when the model is good enough.
Use ensemble and resampling techniques to improve model performance.
The concept of ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the model.
Oversampling is a strategy used where the idea is simple and intuitive: If one class has too few instances in the training set, we choose more instances from that class for training until it’s larger.
+	Class imbalance refers to a situation in which the existing classes in a dataset aren’t equally represented.
Random oversampling is instances where the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.
The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased.
A downside of oversampling with SMOTE is its reliance on the immediate neighbors of a data point. Because the algorithm doesn’t see the overall distribution of data, the new data points it creates can be heavily influenced by outliers. This can lead to noisy data. With downsampling, the downsides are that it involves loss of data and is not an option when the dataset is small. One way to deal with these challenges is to use a sampling strategy that is a combination of oversampling and undersampling.
SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms.
SMOTEENN is a two-step process:
+	
i.	Oversample the minority class with SMOTE.
	
ii.	Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

## Challenge Overview
In this challenge, we built and evaluated several machine learning models to assess credit risk, using data from LendingClub; a peer-to-peer lending services company.

Credit risk is an inherently unbalanced classification problem, as the number of good loans easily outnumber the number of risky loans. Therefore, we employed different techniques to train and evaluate models with unbalanced classes. We used imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. Our final task was to evaluate the performance of these models and made a recommendation on whether they should be used to predict credit risk.

## Challenge Objectives
Implement machine learning models.
Use resampling to attempt to address class imbalance.
Evaluate the performance of machine learning models.
## Challenge Summary
Instructions
You’ll use the imbalanced-learn library to resample the data and build and evaluate logistic regression classifiers using the resampled data. Download the files you’ll need, which include starter code and the dataset:

Download Module -17-Challenge-Resources.zip

You will:

Over sample the data using the RandomOverSampler and SMOTE algorithms.
Undersample the data using the cluster centroids algorithm.
Use a combination approach with the SMOTEENN algorithm.
For each of the above, you’ll:

Train a logistic regression classifier (from Scikit-learn) using the resampled data.
Calculate the balanced accuracy score using balanced_accuracy_score from sklearn.metrics.
Generate a confusion_matrix.
Print the classification report (classification_report_imbalanced from imblearn.metrics).
Lastly, you’ll write a brief summary and analysis of the models’ performance. Describe the precision and recall scores, as well as the balanced accuracy score. Additionally, include a final recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

## Extension
For the extension, you’ll train and compare two different ensemble classifiers to predict loan risk and evaluate each model. Note that you’ll use the following modules, which you have not used before. They are very similar to ones you’ve seen: BalancedRandomForestClassifier and EasyEnsembleClassifier, both from imblearn.ensemble. These modules combine resampling and model training into a single step. Consult the following documentation for more details:

Section 5.1.2. Forest of randomized trees (Links to an external site.)
imblearn.ensemble.EasyEnsembleClassifier (Links to an external site.)
Use 100 estimators for both classifiers, and complete the following steps for each model:

Train the model and generate predictions.
Calculate the balanced accuracy score.
Generate a confusion matrix.
Print the classification report (classification_report_imbalanced from imblearn.metrics).
For the BalancedRandomForestClassifier, print the feature importance, sorted in descending order (from most to least important feature), along with the feature score.
Lastly, you’ll write a brief summary and analysis of the models’ performance. Describe the precision and recall scores, as well as the balanced accuracy score. Additionally, include a final recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

## Submission
Host your challenge assignment on GitHub, including:


## Analysis
Naive Random Oversampling The precision score are 1.00 for predicting low-risk and 0.01 for predicting high-risk. The recall score is about the 0.63 for low-risk and 0.65 for high-risk. The balanced accuracy score is 0.63.

SMOTE Oversampling The precision score are 1.00 for predicting low-risk and 0.01 for predicting high-risk. The recall score is about the same 0.65 for low-risk and 0.61 for high-ris) for both categories. The balanced accuracy score is 0.63, which is pretty low.

Undersampling The precision score are 1.00 for predicting low-risk and 0.01 for predicting high-risk. The recall score is about the same 0.4 for low-risk and 0.63 for high-risk for both categories. The balanced accuracy score is 0.51, which is low.

Combination Sampling The precision score are 1.00 for predicting low-risk and 0.01 for predicting high-risk. The recall score is about the same 0.74 for low-risk and 0.53 for high-risk for both categories. The balanced accuracy score is 0.63, which is pretty low.

Balanced Random Forest Classifier The precision score are 1.00 for predicting low-risk and 0.04 for predicting high-risk. The recall score is about the same 0.67 for low-risk and 0.9 for high-risk for both categories. The balanced accuracy score is 0.78, which is the highest.

Conclusion and Recommendations Looking at all the different models, we observe that Combination Sampling and SMOTE Oversampling performs the same so there is no point to do a Combination Sampling.but having the Balanced Random Forest Classifie model with the highest accurecy score, we would perform more analysis on Balanced Random Forest Classifier model.


Your Jupyter Notebook file(s) with your code and analysis.
Your dataset.
A README.md file describing your project.
