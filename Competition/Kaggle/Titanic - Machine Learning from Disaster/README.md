![](img/header.jpg)

<p>
    <img src="https://img.shields.io/badge/Kaggle-blue">
    <img src="https://img.shields.io/badge/Python-1E90FF?logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/Pandas-696969?logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/Accuracy-%200.8065%20-31bfe2">
</p>
# The Titanic Dataset: A Step-by-Step Guide to Predicting Survival | Jupyter-lab & python script | Kaggle

In this project, we plan to build a model step by step to predict the life conditions of other passengers with the help of machine learning.

This challenge has been published on the Kegel website. [see this challenge on Kaggle](https://www.kaggle.com/competitions/titanic/overview)



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

## Let's Do it

Continue step by step with me.

### Step1: Data Collection

Data collection is the process of obtaining relevant data from various sources for a specific purpose. It involves identifying the sources, gathering the data in various formats, and ensuring its accuracy and completeness. The quality and quantity of the collected data are crucial to the success of machine learning projects.



> 👉 The data was provided by Kegel, so we skip this step.

Link Download from Kaggle: [Link](https://www.kaggle.com/competitions/titanic/data?select=train.csv)

