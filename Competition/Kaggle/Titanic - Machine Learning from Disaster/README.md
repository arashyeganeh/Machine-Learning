![](img/header.jpg)

<p>
    <img src="https://img.shields.io/badge/Kaggle-blue">
    <img src="https://img.shields.io/badge/Python-1E90FF?logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/Pandas-696969?logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/Accuracy-%200.8065%20-31bfe2">
</p>
# The Titanic Dataset: A Step-by-Step Guide to Predicting Survival | Kaggle | 💀 😀

In this project, we plan to build a model step by step to predict the life conditions of other passengers with the help of machine learning.

This challenge has been published on the Kaggle website. [see this challenge on Kaggle](https://www.kaggle.com/competitions/titanic/overview)



## The goal of the challenge

Predicting the status of passengers in terms of whether they are alive or dead.

## Data

There is a file `train.csv` exist that includes 891 records (without features or header).

### Data Dictionary

| **Variable** |               **Definition**               |                    **Key**                     |
| :----------: | :----------------------------------------: | :--------------------------------------------: |
|   survival   |                  Survival                  |                0 = No, 1 = Yes                 |
|    pclass    |                Ticket class                |           1 = 1st, 2 = 2nd, 3 = 3rd            |
|     sex      |                    Sex                     |                                                |
|     Age      |                Age in years                |                                                |
|    sibsp     | # of siblings / spouses aboard the Titanic |                                                |
|    parch     | # of parents / children aboard the Titanic |                                                |
|    ticket    |               Ticket number                |                                                |
|     fare     |               Passenger fare               |                                                |
|    cabin     |                Cabin number                |                                                |
|   embarked   |            Port of Embarkation             | C = Cherbourg, Q = Queenstown, S = Southampton |

### Variable Notes

**pclass**: A proxy for socio-economic status (SES)
`1st `= Upper
`2nd `= Middle
`3rd `= Lower

**age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

**sibsp**: The dataset defines family relations in this way...
`Sibling `= brother, sister, stepbrother, stepsister
`Spouse `= husband, wife (mistresses and fiancés were ignored)

**parch**: The dataset defines family relations in this way...
`Parent `= mother, father
`Child `= daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

## Requirements

* Install the last stable version of `Python`

  [Download Python](https://www.python.org/downloads/)

* Install `JupyterLab` (optional)

  Scripts can be written in any IDE, but `JupyterLab ` makes it easier.

  ```bash
  pip install jupyterlab
  ```

* Install libraries required by this project using pip.

  ```bash
  pip install pandas seaborn scikit-learn lazypredict
  ```

## Let's Do it 💪

Continue step by step with me.

### Step1. Data Collection

Data collection is the process of obtaining relevant data from various sources for a specific purpose. It involves identifying the sources, gathering the data in various formats, and ensuring its accuracy and completeness. The quality and quantity of the collected data are crucial to the success of machine learning projects.

👉 Link Download from Kaggle: [Link](https://www.kaggle.com/competitions/titanic/data?select=train.csv)

> 📚 pandas: for data manipulation and analysis.

```python
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```

### Step2. Data Exploration

Data exploration is the crucial step of analyzing and understanding a dataset to identify patterns, relationships, and trends. It involves summarizing, visualizing, feature engineering, hypothesis testing, and identifying data quality issues. This iterative process helps ensure the data is properly understood and prepared for building more accurate and effective machine learning models.

By using `info()` I can find out how many columns there are, and what their data types are.

> 📚 seaborn: for data visualization

```python
train.info()
```

```bash
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```

There're 12 columns

❔ How many missing values are there in each column?

```python
train.isna().sum()
```

```
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```

The columns `Age`, `Cabin`, and `Mounted` have missing values.

By using `head()` can see the top 5 first rows of data frame.

```python
train.head()
```

![](img/train_df.head.PNG)

```python
import seaborn as sns
sns.countplot(x='Survived', data=train)
```

![](img/countplot_survived.PNG)

```python
sns.countplot(x='Survived', hue='Sex', data=train)
```

![](img/countplot_survived_sex.PNG)

| **Column**  | **Missing values** |                         **Describe**                         | Action |
| :---------: | :----------------: | :----------------------------------------------------------: | :----: |
| PassengerId |         0          |          It is not providing any useful information          |   👎    |
|  Survived   |         0          |                              -                               |   ✔️    |
|   Pclass    |         0          |                              -                               |   ✔️    |
|    Name     |         0          |          It is not providing any useful information          |   👎    |
|     Sex     |         0          | The values in this column are currently defined as strings and must be converted into a numerical format for the machine to analyze |   ⚠️    |
|     Age     |        177         |                  it includes missing value                   |   ⚠️    |
|    SibSp    |         0          |                              -                               |   ✔️    |
|    Parch    |         0          |                              -                               |   ✔️    |
|   Ticket    |         0          |          It is not providing any useful information          |   👎    |
|    Fare     |         0          |                              -                               |   ✔️    |
|    Cabin    |        687         | This column has the highest number of missing values in the dataset, and it cannot be correlated with any other features. |   👎    |
|  Embarked   |         2          |          It is not providing any useful information          |   👎    |

* 👎  `It means useless so will delete`
* ✔️ `It means useful`
* ⚠️ `It means Useful but data need to normalize`

### Step3. Data Cleaning

Data cleaning is the process of identifying and correcting errors, inconsistencies, and inaccuracies in a dataset. It involves identifying missing data, removing duplicates, standardizing data, correcting errors, handling outliers, and ensuring data consistency. This iterative process is crucial for ensuring the accuracy and performance of machine learning models.

```python
def DropUselessCol(df):
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

def FillnaAge(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    
def CleanData(df):
    DropUselessCol(df)
    FillnaAge(df)
    
CleanData(train)
train.head()
```

![](img/data_cleaning.PNG)

### Step4. Feature Engineering & Data Preprocessing

Feature engineering is the process of selecting and transforming features in a dataset to create new, more meaningful features that improve the performance of machine learning models. This involves feature selection, extraction, scaling, encoding, and dimensionality reduction. It's an iterative process that can improve the accuracy and effectiveness of machine learning models.

```python
def convertToNumeric(df):
    sex = pd.get_dummies(df['Sex'], drop_first=True)
    df.drop(['Sex'], axis=1, inplace=True)
    df = pd.concat([df, sex], axis=1)

def castColToInt(df, name):
    df[name] = df[name].astype(int)
    
def FeatureEngineering(df):
    convertToNumeric(df)
    castColToInt(df, 'Age')
    castColToInt(df, 'Fare')
    
FeatureEngineering(train)
print(train)
```

![](img/feature_engineering.PNG)

### Step5. Data Splitting

Data splitting is a crucial step in machine learning that divides the dataset into training and testing sets. The training set is used to train the model, and the testing set is used to evaluate its performance on new, unseen data. Common approaches include the hold-out method, cross-validation, and leave-one-out method. Data splitting should be done randomly and in a way that preserves the distribution of the data, which can be achieved by stratified sampling.

![](img/data_spliting.png)

![](img/data_spliting_train.png)

> 📚 sklearn (scikit-learn): for machine learning algorithms

```python
from sklearn.model_selection import train_test_split

X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Step6. Model Selection

Model selection is about selecting the best machine learning model for a problem, considering the model architecture, hyperparameters, and optimization algorithm. Different machine learning models, such as decision trees, support vector machines, and neural networks, have their strengths and weaknesses. Model selection involves defining the problem, selecting a performance metric, choosing candidate models, splitting the data, training and evaluating models, selecting the best model based on the validation set, fine-tuning the selected model, and optimizing its performance. It is crucial to choose a model that is suitable for the problem and can generalize well to new data.

[Documentation: lazypredict](https://lazypredict.readthedocs.io/en/latest/)

### Step7. Model Training

Model training is the process of adjusting a machine learning model's parameters to fit the training data and make predictions. It involves data preparation, model selection, initialization, training, validation, evaluation, and fine-tuning. The goal is to minimize the loss function and improve the model's ability to generalize to new data. Preprocessing data and selecting appropriate hyperparameters are critical for successful model training.

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
```

### Step8. Evaluation

Evaluation assesses the performance of a machine learning model on new data. Metrics like accuracy, precision, recall, F1-score, AUC-ROC, MSE, and MAE are used to evaluate a model's performance. The dataset is split into training, validation, and test sets. Choosing an appropriate metric and tuning hyperparameters are crucial for successful evaluation.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

predict = logreg.predict(X_test)

accuracy = accuracy_score(y_test, predict)
precision = precision_score(y_test, predict)
recall = recall_score(y_test, predict)
f1 = f1_score(y_test, predict)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
```

> Accuracy: 0.71
> Precision: 0.72
> Recall: 0.45
> F1 Score: 0.55

### Step9. Hyperparameter Tuning

Hyperparameter tuning finds the optimal set of hyperparameters for a machine learning model, which control the model's behavior during training. Examples of hyperparameters include learning rate, regularization strength, and number of layers/neurons. Hyperparameter tuning can be done through methods like grid search, random search, Bayesian optimization, and genetic algorithms. Successful tuning involves careful selection of hyperparameter range, appropriate search method, and evaluation on a validation set to avoid overfitting. Computational resources should also be considered.

```
from sklearn.model_selection import GridSearchCV

hyperparameters = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

grid = GridSearchCV(logreg, hyperparameters, cv=5)
grid.fit(X_train, y_train)

tuning_predict = grid.predict(X_test)
tuning_accuracy = accuracy_score(y_test, tuning_predict)
tuning_precision = precision_score(y_test, tuning_predict)
tuning_recall = recall_score(y_test, tuning_predict)
tuning_f1 = f1_score(y_test, tuning_predict)

print("Accuracy: {:.2f}".format(tuning_accuracy))
print("Precision: {:.2f}".format(tuning_precision))
print("Recall: {:.2f}".format(tuning_recall))
print("F1 Score: {:.2f}".format(tuning_f1))
```

### Step10. Model Deployment

```
test_pred = logreg.predict(test_df)

submission = pd.DataFrame({
    'PassengerId': pd.read_csv('test.csv')['PassengerId'],
    'Survived': test_pred
})

submission.to_csv('submission.csv', index=False)
```