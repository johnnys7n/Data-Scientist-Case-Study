<img src="https://seeklogo.com/images/R/Roche-logo-A80FCF9553-seeklogo.com.png" style="width:120px;height:70px">

# Roche Data Science - Home Study Case

By: Johnny Sin
Date: July 7-10, 2023

# Outline of the Project

## 1. Project Information
Scenario: Working as a Data Scientist and asked by Digital Transformation Managing Partner Simon to start a project that uses data collected on stroke patients over the past few years.

**About this Company:**
* Mission: developing innovative treatments for stroke, with an average of 11K patients helped annually.

## 2. Describing the data providing

Feature Information:
1. ID: Unique Identification Number
2. Gender: Male / Female / Other (Categorical Variable)
3. Age_In_Days: Indicates patient's age in days (numerical variable)
4. Hypertension: 1 - has hypertension | 0 - no hypertension (binomial variable)
5. Heart_Disease: 1 - has heart disease | 0 - no heart disease (binomial variable)
6. Ever_married: Yes - patient is / was ever married | No - patient has never been married (binomial variable)
7. Type_Of_Work: working status, 'self-employed', 'works in private firm', 'government job', 'still a child' (categorical variable)
8. Residence: Urban - patient currently lives in urban | Rural - patient lives in rural area (binomial variable)
9. Avg_Glucose: patient's average glucose level for the past 3 months
10. BMI: patient's current BMI score
11. Smoking_Status: indication of smoking habits (categorical variable)
12. **TARGET** Stroke: 1 - patient after stroke | 0 - no stroke (binomial variable)

Already looking at the features of the dataset, I will be using various classification algorithms for this project

## 3. Exploratory Data / Statistical Analysis

Here I will be looking at the various data descriptions and also perform minor feature engineering, along with figuring out the autocorrelative/multicollinearity of the features. 

### 3.1 Importing the tools / libraries
We will be using the standard libraries for data science / machine learning in Python
1. Pandas 
2. NumPy
3. MatplotLib
4. Statsmodel
5. Scipy
6. Seaborn
7. Sklearn
8. XGBoost

### 3.2 Checking out the data information and more details
Here we are getting a high level overview of the data to see what needs to be done on the raw data first before plugging it into a modeling pipeline.

We also create lists of column names based on their dtype `object` , `float`, or `int`, in order to be more efficient in our encoding phase later on.

### 3.3 Encoding the Categorical features into Int64 values
we will create a new dataframe called `df_tmp` only for visualization and preliminary statistics reasons. We will manually encode and map numerical values to the categorical features

### 3.4 Checking for Null values
Only two columns `BMI` and the `Smoking Status` columns have missing values. Because the `BMI` column only has 3% missing, we can just simply remove those missing values. And for the `Smoking_Status` it has 30% missing values so we need to create a new column called 'missing or simply call it 3'

### 3.5 Data Visualization 
1. Using a Correlation Matrix to see a high level overview of the feature correlations. And combining that with a DataFrame containing the coefficient values with a gradient style that matches the correlation matrix.
2. Creating a function that visualizes either `value_counts` or `hist` output depending on the categorical `dtype`

#### 3.5.1 `Age_In_Days` column seems to need further processing
Looking at the histogram we can see a immense right-skewness of data, which can be due to an outlier. Looking more into this feature, we see that the average age is ~41 years, which makes sense, however, the max age is 724 years, which is not possible. Using the Tukey's rule, I limited the rows that defied the outlier boundaries. **After removing these data points, I still had 96% of the original samples to work with**

#### 3.5.2 Class distributions
Checking the `Target` column to see if our dataset is class-imbalanced. We see there are `41290` 0-values to `643` 1-values, which is heavily imbalanced. In order to minimize false accuracy errors during our modeling/evaluation phase I need to randomly undersample the majority class to make sure it is closer to the minority sample size. 

After randomly sampling the majority class, I am left with `700` 0-value and `643` 1-values

#### 3.5.3 Looking at the numerical (`dtype = float`) features and their correlations
1. Created a Pairplot to look at the KDE and scatterplot distribution of the numerical features relationships while also coloring the points by stroke-positivity to see any initial clustering possibilities
	* Trends I noticed. Stroke patients seem to be older, with a bit more average glucose, however with no difference in BMI. 
2. I then created another correlation matrix with just the numerical features and see that the the coefficient is on the lower range. 
3. Then, I created a function called `feature_variability` that inputs the 2 columns of interest and the dataframe and outputs a scatterplot and a OLS regression statistical output. 
	* seeing how all these relationship had a $Durbin-Watson$ score close to 2, we can safely assume there is no autocorrelation between the features.
4. Hypothesis testing of Age_In_Days feature 
	* through randomly sampling 50 patients per group to see significant differences in age. 
	* **Conclusion** Hypothesis testing: After randomly sampling 50 from each group The results from this test rules out our null hypothesis and shows a significant difference in the age in the stroke sample group.


#### 3.5.4. Categorical Visualizations in relation with the Target Feature
Here I simply created a plot of subplots that visualized the count of each categorical values and colored `Target` feature. This will allow us to see the distribution of data that we will feed into our modeling pipeline.

#### 3.5.5 Potential Effects of 'Type_Of_Work' on Stroke
As mentioned in the Overview of Stroke slide in the presentation, work-life stress may be associated with higher levels of stroke incidence. So here I investigate the potential relationships that Type_Of_Work might have on Heart_Disease, BMI, and Hypertension features, all of which were associated with higher risk of stroke. Jobs 2 and 4 seems to be less associated with hypertension and heart disease (both of which have a higher mean of incidence in patients with stroke)
 
>Here instead of using the `df_tmp` dataset that has been randomly resampled, I will use the df_many to capture a more comprehensive understanding.

Recall the encoding: 
* Private: 0
* Self-employed: 1
* children: 2
* Govt_job: 3
* Never_worked: 4
    
2 and 4 seems to be consistently low on both Heart Disease and Hypertension and even in BMI.

## 4. Modeling: Finding the Best Estimator
For the modeling portion of this case study, I decided to focus on screening through these 6 estimators:
1. Logistic Regression
2. Random Forest Classifier
3. KNeighbors Classifier
4. SVC
5. XGBoost
6. DecisionTree
7. Naive Bayes

### 4.1 Getting the data ready for modeling
First I created a function that will prepare any data with the similar column structure as our training data into a cleaned data: (`preparing_data()` function below is basically a compilation of all the steps that were involved in preprocessing up until now. This will be for future scenarios where it might be a good idea to re-evaluate the model with new data or model suggestions.)
1. creating a copy of the working dataframe
2. removing columns with the BMI missing values
3. removing the ID column as well 
4. renaming the `Stroke` as `Target`
5. detecting outliers and removing outliers in the `Age_In_Days` variable
6. Encoding the `Smoking_Status` column and filling missing value with 'Missing' as a third category
7. Undersampling the Majority class so balance out the classes

### 4.2 Splitting our Dataset into Training and Testing
Using the `train_test_split` with the parameter `test_size = 0.2` to split the dataset into training and testing

### 4.3 Creating our Pipeline
Here I will be creating a modeling Pipeline with a ColumnTransformer to impute and encode the dataframe. 

Recall we have separate lists of columns:
1. `int_features` (already encoded)
2. `float_features` (no need for encoding)
3. `cat_features` (need encoding)

First we need to remove the `ID` and `Stroke` columns from the `int_features`

The two commented-out lines of code for the other categorical features (`Residence`, `Type_Of_Work`, and `Ever_Married`) can be used later on when fine tuning and feature engineering for future models.

Then I created a `ModelEvaluator` class that can:
1. preprocess the data with `preprocessor()`
2. model the data with `modeler()`
3. score the model with `model_scorer()`
4. print a classification report with `model_report()`

### 4.5 Testing and Evaluating Our Proposed Models
Here I created a dictionary of models to quickly test and used the `ModelEvaluator` class to store the scores and reports to a variable.

For this Class `ModelEvaluator`, the proposed attribute functions / methods are:
1. preprocessor(int_features, cat_features, float_features) --Preprocess Step
2. model_piper(model) -- Modeling Step
3. model_fit() -- Modeling Step
4. model_scorer() -- Modeling Step
5. model_report() -- Modeling Step
6. plot_roc_curve() -- Metrics Evaluation Step
7. crossval_score(self, X, y, cv=5) -- Metrics Evaluation Step
8. plot_confusion_matrix() -- Metrics Evaluation Step
9. get_params() -- Parameter Evaluation Step

This Class will be used for testing the models in a quick and succint way

I also functionizing some of the methods in the `ModelEvaluator` class to make it compatible with hyperparameter-tuned gridsearch models
1. model_scorer(model, X_test, y_test)
2. model_report)model, X_test, y_test)
3. plot_roc_curve(model, X_test, y_test)
4. plot_confusion_matrix(model, X_test, y_test)

Seeing how the RandomForest although has the highest precsion score, we are more interested in the Recall and F1 score which the SVC and LogisticRegression has the highest in those metrics.

Model Summary: (Ranked in recall / f1 score)
1. **SVC** (go into hyperparameter tuning for more investigation)
2. **LogisticRegression** (go into hyperparameter tuning for more investigation)
3. **RandomForest** (go into hyperparameter tuning for more investigation)
4. KNN (stop for now)
5. XGBoostClassifier (stop faor now)
6. DecisionTree (stop for now)
7. Naive Bayes (stop for now)

## 5. Hyperparameter Tuning:
### 5.1 Testing SVC Model performance after Hyperparameter Tuning
I will use our `ModelEvaluator` class to fit our SVC model on the training dataset

#### 5.1.1 Re-instantitating the ModelEvaluator on SVC for recording inital metrics output
I  first wanted to see the inital scores / metrics for the SVC model using the confusion matrix and the classification report to note the starting metrics to improve from. 
#### 5.1.2 Hypertuning the SVC Model with Grid Search
Then I created a grid for the GridSearchCV class. 

### 5.2. Testing Logistic Regression Model performance after Hyperparameter Tuning
#### 5.2.1 5.2.1 Re-instantitating the ModelEvaluator on LogisticRegression for recording inital metrics output
#### 5.2.2 Hypertuning the Logistic Regression with Grid Search

### 5.3 Testing RandomForest Classifier performance after Hyperparameter Tuning
### 5.3.1 Re-instantitating the ModelEvaluator on RandomForest for recording inital metrics output

## 6. Feature Importance and Summary:

Using the permutation_importances functionm, which is a model-agnostic measurement (although not a predictive indicator, it does output a value of its importance in terms of model error improvement), showed that `Age_In_Days` is very important to the fitting of the model. `RandomForestClassifier` also seem to have quite the differing importances when compared to the other two. 

Steps that I have completed for this project includes:
1. Data Cleaning and Wrangling
2. Feature Engineering
3. Exploratory Data and Statistical Analysis
4. Model Evaluation
5. Hyperparameter Tuning
6. Feature Importance

### Recommendations:
**Recommendation 1**: Because the SVC model had the highest starting metrics (in recall particularly because this measures the ability to find all relevant instances of a class) it might be worthwhile to experiment with this model further and tune its hyperparameters. 

**Recommendation 2**: However, if no improvement is seen with SVC, it might also be worthwhile to look at the RandomForestClassifier, because this model had the greatest change during the first phase of hyperparameter tuning (4% increase in recall)


### Next Steps:
The modeling portion of this project was very exciting and fruitful. Having the opportunity to work with data for the proposed goal of stroke diagnosis and potentially improving health care was a rewarding experience.

Because the experimentation and machine learning optimization for classification problems portions of this assignment can be a cyclic process of repeated improving and tweaking, there are many steps that I have yet to explore due to the project time constraint.

1. Optimize the model further with a broader range of values for Grid Search to optimize these three models first, then potentially include the other models, as well as models that were not included in the list. 
    * With the way this notebook is structured, it will be simple to go back and experiment with other choices (changing the way the unbalanced classes are handled and/or removing columns that are deemed irrelevant).
    * Consider even including new features (such as Race/Ethnicity or Family history) either through changing collection methods or feature engineering: ie. Considering that Type_Of_Work might be a variable that measure one modality of stress and use that information to either create a new feature or design a feature through merging multiple correlated columns.
    * In order to more accurately capture the distribution of class data it might be also important to optimize the model on the imbalanced class data (original) and emphasize the class_weights parameter for each model, since this more accurately resembles the real-world situation.

2. After optimizing the model further using the proposed methods, and perhaps after discussion with experts / client, it would be great to deploy this model using a web application framework such as Flask or Django or even in Shiny to make this model more accessible for hospital use during stroke treatment recommendation for patients.

