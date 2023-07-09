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

Here I will be looking at the various data descriptions and also feature engineering, along with figuring out the autocorrelative/multicollinearity of the features. 

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
Looking at the histogram we can see a immense right-skewness of data, which can be due to an outlier. Looking more into this feature, we see that the average age is ~41 years, which makes sense, however, the max age is 724 years, which is not possible. Using the Tukey's rule, I limited the rows that defied the outlier boundaries. **After removing these data points, I still had 96% of the original samples to work with **

#### 3.5.2 Class distributions
Checking the `Target` column to see if our dataset is class-imbalanced. We see there are `41290` 0-values to `643` 1-values, which is heavily imbalanced. In order to minimize false accuracy errors during our modeling/evaluation phase I need to randomly undersample the majority class to make sure it is closer to the minority sample size. 

After randomly sampling the majority class, I am left with `700` 0-value and `643` 1-values

#### 3.5.3 Looking at the numerical (`dtype = float`) features and their correlations
1. Created a Pairplot to look at the KDE and scatterplot distribution of the numerical features relationships while also coloring the points by stroke-positivity to see any initial clustering possibilities
	* Trends I noticed. Stroke patients seem to be older, with a bit more average glucose, however with no difference in BMI. 
2. I then created another correlation matrix with just the numerical features and see that the the coefficient is on the lower range. 
3. Then, I created a function called `feature_variability` that inputs the 2 columns of interest and the dataframe and outputs a scatterplot and a OLS regression statistical output. 
	* seeing how all these relationship had a $Durbin-Watson$ score close to 2, we can safely assume there is no autocorrelation between the features.
4. Lastly, I then 

#### 3.5.4. Categorical Visualizations in relation with the Target Feature
Here I simply created a plot of subplots that visualized the count of each categorical values and colored `Target` feature. This will allow us to see the distribution of data that we will feed into our modeling pipeline.

## 4. Modeling: Finding the Best Estimator
For the modeling portion of this case study, I decided to focus on screening through these 6 estimators:
1. Logistic Regression
2. Random Forest Classifier
3. KNeighbors Classifier
4. SVC
5. Linear SVC
6. XGBoost
7. DecisionTree
8. Naive Bayes

### 4.1 Getting the data ready for modeling
First I created a function that will prepare any data with the similar column structure as our training data into a cleaned data:
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

Then I created a `ModelEvaluator` class that can:
1. preprocess the data with `preprocessor()`
2. model the data with `modeler()`
3. score the model with `model_scorer()`
4. print a classification report with `model_report()`

### 4.5 Testing and Evaluating Our Proposed Models
Here I created a dictionary of models to quickly test and used the `ModelEvaluator` class to store the scores and reports to a variable.

Seeing how the RandomForest although has the highest precsion score, we are more interested in the Recall and F1 score which the SVC and LogisticRegression has the highest in those metrics.

Model Summary: (Ranked in recall / f1 score)
1. **SVC** (go into hyperparameter tuning for more investigation)
2. **LogisticRegression** (go into hyperparameter tuning for more investigation)
3. LinearSVC (stop for now)
4. RandomForest (stop for now)
5. KNN (stop for now)
6. XGBoostClassifier (stop for now)
7. DecisionTree (stop for now)
8. Naive Bayes (stop for now)

## 5. Hyperparameter Tuning:
### 5.1 Testing SVC Model performance after Hyperparameter Tuning
I will use our `ModelEvaluator` class to fit our SVC model on the training dataset

#### 5.1.1 Creating a GridSearchCV on the parameters
I  first wanted to see the inital scores / metrics for the SVC model using the confusion matrix and the classification report to note the starting metrics to improve from. 

Then I created a grid for the GridSearchCV class. 

### 5.2. Testing Logistic Regression Model performance after Hyperparameter Tuning
#### 5.2.1 Obtaining the 'Pre-Tuned' metrics of the Logistic Regression
#### 5.2.2 Hypertuning the Logistic Regression with Grid Search

## 6. Feature Importance and Summary:

Looking at the Feature importances from both the SVC and LogisticRegression we can see that surprisingly `Type_Of_Work` has the highest coefficient, followed by `Residence` and `Ever_Married`. 

**Summary** This project required an optimization of a classifier in order to predict whether a patient has stroke or not. After data manipulation to understand the structure of the data, missing values, class imbalances, necessary encoding, I then experimented with various machine learning models. SVC and LogisticRegression had the highest recall score along with 

