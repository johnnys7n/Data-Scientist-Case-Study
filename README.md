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
### 3.2 Checking out the data information and more details
Here we are getting a high level overview of the data to see what needs to be done on the raw data first before plugging it into a modeling pipeline

### 3.3 Encoding the Categorical features into Int64 values
we will create a new dataframe called `df_tmp` only for visualization and preliminary statistics reasons. We will manually encode and map numerical values to the categorical features

### 3.4 Checking for Null values
Only two columns `BMI` and the `Smoking Status` columns have missing values. Because the `BMI` column only has 3% missing, we can just simply remove those missing values. And for the `Smoking_Status` it has 30% missing values so we need to create a new column called 'missing or simply call it 3'

### 3.5 Data Visualization 
1. Using a Correlation Matrix to see a high level overview of the feature correlations 
2. Pairplot analysis for looking at potential clustering of data points
3. Creating a function that visualizes either `value_counts` or `hist` output depending on the categorical `dtype`

### 3.5.1 `Age_In_Days` column seems to need further processing
Looking at the histogram we can see a immense right-skewness of data, which can be due to an outlier. Looking more into this feature, we see that the average age is ~41 years, which makes sense, however, the max age is 724 years, which is not possible. Using the Tukey's rule, I limited the rows that defied the outlier boundaries. **After removing these data points, I still had 96% of the original samples to work with **
### 3.5.2 Looking at the numerical features and their correlations
Then, I investigated the two numerical columns, `Age_In_Days` and `Avg_Glucose`. Plotting the first 2000 of these two features as a scatterplot showed no correlation that can be explained with linear regression. Although there might be some clustering, we will save those analyses for later.  

## 4. Finding the Best Estimator
For the modeling portion of this case study, I decided to focus on screening through these 5 estimators:
1. Logistic Regression
2. Random Forest Classifier
3. KNeighbors Classifier
4. SVC
5. Linear SVC

### 4.1 Getting the data ready for modeling
