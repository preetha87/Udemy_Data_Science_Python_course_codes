
# coding: utf-8

# In[1]:

#Exercise to depict the fact that Ensemble methods like Random Forest and Gradient Boosting are better than Logististic Regression!
import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[2]:

#Read in the training set
titanic = pd.read_csv('train set data - Titanic.csv')


# In[3]:

titanic['Survived'].unique()


# In[7]:

#Read in the test set
titanic_test = pd.read_csv('test set data - Titanic.csv')


# In[8]:

#Information on the training data set
titanic.info()
#Cabin has 77% of the data missing
#Need to extract cabin letters and categorize into A,B,C,D,E, etc.
#Age has 19% of the data missing
#Embarked has two observations missing
#Looks like these values are missing at random!
#Can't use imputation for cabin - it's qualitative data


# In[9]:

#Get the info on the test set
titanic_test.info()
#21% of the values for age variable are missing
#No missing values for Embarked
#1 missing value for Fare


# In[10]:

#Impute missing values for Age with the median value - training set
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median)


# In[11]:

#Convert the sex column into numeric - 1 for male and 0 for female
titanic.loc[titanic['Sex']=='male', 'Sex'] = 1
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1


# In[12]:

#Get a count of the number of values in the Embarked column
titanic['Embarked'].value_counts()


# In[13]:

#Utilize common case imputation - since Southampton is the most common port of Embark
#Replace the null values with the value S for Southampton
titanic['Embarked'] = titanic['Embarked'].fillna('S')


# In[14]:

titanic['Embarked'].unique()


# In[15]:

titanic_test['Embarked'].unique()


# In[16]:

#Convert the embarked column - code 1 for Southampton, 2 for Queenstown and 3 for Cherobourgh
titanic.loc[titanic['Embarked']=='S', 'Embarked'] = 1
titanic.loc[titanic['Embarked']=='Q', 'Embarked'] = 2
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 3


# In[17]:

#Mimic most of these changes in the test set as well! Fill in the missing value for Fare with the median fare value!
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median)
titanic_test.loc[titanic_test['Sex']=='male', 'Sex'] = 1
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic['Fare'].median)
titanic_test.loc[titanic_test['Embarked']=='S', 'Embarked'] = 1
titanic_test.loc[titanic_test['Embarked']=='Q', 'Embarked'] = 2
titanic_test.loc[titanic_test['Embarked'] == 'C', 'Embarked'] = 3


# In[18]:

#Feature Engineering - generating some interesting features which could lend predictive power to the dependent variable
#Create a family size column by adding the SibSp and Parch columns together
#Parch -- The number of parents and children the passenger had on board.
#SibSp - The number of siblings and spouses the passenger had on board.
titanic['Family Size'] = titanic['Parch'] + titanic['SibSp']
titanic_test['Family Size'] = titanic_test['Parch'] + titanic_test['SibSp']


# In[19]:

#How wealthy a passenger is, depends on the title of that passenger. 
#Get a function to extract the title
import re

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
# Verify that we converted everything.
print(pd.value_counts(titles))

# Add in the title column.
titanic["Title"] = titles


# In[20]:

#Do the same for the test set
titles = titanic_test["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
# Verify that we converted everything.
print(pd.value_counts(titles))

# Add in the title column.
titanic_test["Title"] = titles


# In[21]:

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Family Size", "Title"]

# Perform feature selection - select the 4 best features
selector = SelectKBest(f_classif, k=4)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)


# In[22]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score


# In[27]:

predictors = ["Pclass", "Sex", "Fare", "Title"]
#min_samples_split : This should be around 0.5% - 1% of total values. 
#min_samples_leaf : Can be selected based on intuition. This is just used for preventing overfitting 
#and again a small value because of imbalanced classes.
#max_depth : Should be chosen (5-8) based on the number of observations and predictors. 
#max_features : ‘sqrt’ : Its a general thumb-rule to start with square root.
alg = RandomForestClassifier(n_estimators=50, min_samples_split=8, min_samples_leaf=4)
# Compute the accuracy score for all the cross validation folds. This is k-fold cross validation. The number of folds = 3
scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# In[25]:

from sklearn.ensemble import GradientBoostingClassifier


# In[30]:

alg2 = GradientBoostingClassifier(n_estimators=50, min_samples_split=8, min_samples_leaf=4, max_depth=5)
scores = cross_val_score(alg2, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())


# In[28]:

#The accuracy scores for the Gradient Boosting is 0.815 and for the Random Forest classifier is 0.809


# In[29]:

from sklearn.linear_model import LogisticRegression


# In[31]:

alg3 = LogisticRegression()
score = cross_val_score(alg3, titanic[predictors], titanic['Survived'], cv=3)
print(score.mean())
#The accuracy score for the Logistric Regression algorithm  is just 0.73!


# In[33]:

#Train the algorithm using all the training data
#Run the Random Forest algorithm
alg.fit(titanic[predictors], titanic["Survived"])
#Make predictions on the test set
predictions1 = alg.predict(titanic_test[predictors])


# In[34]:

#Train the algorithm using all the training data
#Run Gradient Boosting algorithm
alg2.fit(titanic[predictors], titanic["Survived"])
#Make predictions on the test set
predictions2 = alg.predict(titanic_test[predictors])


# In[ ]:



