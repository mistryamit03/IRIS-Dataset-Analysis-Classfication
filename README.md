


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


**Loading and Reading the dataset**

df = pd.read_csv('Iris.csv')

**To delete the row**

df = df.drop(columns= ['Id'])
df.head()

**To display stats about data**

df.describe()

**To get basic info about the data type**

df.info()

**To display number of samples on each class**

df['Species'].value_counts()


**PREPROCESSING THE DATA SET**
Check for Null Values
While training the model we need to remove all the null values
The following code will execute sum of null values for the corresponding columns

df.isnull().sum()


**Exploratory Data Analysis: Display basic graphs**

df['SepalLengthCm'].hist()
df['SepalWidthCm'].hist()
df['PetalLengthCm'].hist()
df['PetalWidthCm'].hist()

**Scatter Plot**

colors = ['red', 'orange', 'blue']
species = ['Iris-setosa','Iris-versicolor','Iris-virginica']


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label = species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()


    for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'], c = colors[i], label = species[i])
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()


**Correlation Matrix: A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. 
The value is in the range of -1 to 1. If two variables have high correlation, we can neglect one variable from those two.**

adf.corr()
corr = df.corr()
fig, ax = plt.subplots(figsize = (4,8))
sns.heatmap(corr, annot = True, ax=ax, cmap = 'coolwarm')

**Label Encoder: In ML, we usually deal with datasets which contains multiple lables in one or more than one columns. 
These labels can be in the form of words or numbers.Label encoding refers to converting the lables into numeric form so as to convert it into the MR form.**

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head()


**Model Training**

from sklearn.model_selection import train_test_split
X = df.drop(columns = ['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


