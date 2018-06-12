# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 08:48:12 2018

@author: LENOVO
"""

#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#importing training and testing datasets
traindf = pd.read_csv("train.csv")
testdf = pd.read_csv("test.csv")
print("the size of training set:", traindf.shape)
print("the size of test set:", testdf.shape)

#visualising the datasets
traindf.head(20)
testdf.head(20)
traindf.describe

#examining the target variable
sns.distplot(traindf['is_pass'],kde = False)#imbalenced class problem 
#missing values
def missing_values_table(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_values_table(traindf)
missing_values_table(testdf)
#lets fill the missing coloumns 1st 
#for replacing with mean or median or mode lets see the disctibution of age
traindf['age'] = traindf['age'].astype(int)
sns.distplot(traindf['age'].dropna(),kde = False)
sns.distplot(testdf['age'].dropna(),kde = False)
traindf['age'] = traindf['age'].fillna(traindf['age'].median())
testdf['age'] = testdf['age'].fillna(testdf['age'].median())
traindf['trainee_engagement_rating'] = traindf['trainee_engagement_rating'].fillna(traindf['trainee_engagement_rating'].mode()[0])
testdf['trainee_engagement_rating'] = testdf['trainee_engagement_rating'].fillna(testdf['trainee_engagement_rating'].mode()[0])
#exploratory data anlysis
#let us see who is more in dataset variable wise
sns.countplot(x = 'gender',data = traindf,color = 'blue')#male was high
sns.countplot(x = 'is_handicapped',data = traindf,color = 'blue')#only few are hadicapped
sns.countplot(x = 'education',data = traindf,color = 'blue')#most of the candidates from school diploma
sns.countplot(x = 'program_id',data = traindf,color = 'blue')#most are from y_1 and y_3
sns.countplot(x = 'program_type',data = traindf)#y series are more
sns.barplot(x = 'difficulty_level' , y = 'is_pass',hue = 'gender',data = traindf)#almost male has dominace
sns.barplot(x = 'education',y = 'is_pass',hue = 'gender',data = traindf)
sns.barplot(x = 'gender' , y = 'is_pass',hue = 'is_handicapped',data = traindf)
sns.factorplot('trainee_engagement_rating','is_pass',data = traindf,aspect = 2.5)#the more he engaged the more he had chance to pass
sns.factorplot('education','trainee_engagement_rating',data = traindf,aspect = 2.5)#who was educated has engaged more and have high chance to pass
sns.factorplot('program_type','is_pass',data = traindf,aspect = 2.5)#x guys has more pass percent
sns.factorplot('program_id','is_pass',data = traindf,aspect = 2.5)#x and y guys are performing good
sns.barplot(x = 'city_tier' , y = 'education' ,data = traindf)#good to see that 
sns.barplot(x = 'city_tier' , y = 'is_pass' ,data = traindf)#citytier of 1 are having good pass as there are more masters degree holder our analysis is tru
sns.barplot(x = 'city_tier' , y = 'program_type' ,data = traindf)
#now creating dummies
traindf=traindf.drop(['id', 'trainee_id', 'test_id'], axis=1)
testdf=testdf.drop(['id', 'trainee_id', 'test_id'], axis=1)
#creating features and target
features = traindf.drop(['is_pass'],axis = 1)
target = traindf['is_pass']
features=pd.get_dummies(features)
traindf=pd.get_dummies(traindf)
testdf=pd.get_dummies(testdf)




