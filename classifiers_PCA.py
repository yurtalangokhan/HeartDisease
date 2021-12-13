# data analysis, splitting and wrangling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# visualization
import matplotlib.pyplot as plt
import seaborn as sns


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# PCA
from sklearn.decomposition import PCA


print("*****************************************")
# column names in accordance with feature information
col_names = ['age','sex','chest_pain','blood_pressure','serum_cholestoral','fasting_blood_sugar', 'electrocardiographic',
             'max_heart_rate','induced_angina','ST_depression','slope','no_of_vessels','thal','diagnosis']
color = ["#58a3bc","#666666"]
# read the file
dataset = pd.read_csv("processed.cleveland.data", names=col_names, header=None, na_values="?")

print("Number of records: {}\nNumber of variables: {}".format(dataset.shape[0], dataset.shape[1]))


print("*****************************************")
# extract numeric columns and find categorical ones
numeric_columns = ['serum_cholestoral', 'max_heart_rate', 'age', 'blood_pressure', 'ST_depression']
categorical_columns = [c for c in dataset.columns if c not in numeric_columns]
print(categorical_columns)

# count values of explained variable
dataset.diagnosis.value_counts()

# categorize diagnosis values (True=1, False=0)
dataset.diagnosis = (dataset.diagnosis != 0).astype(int)
dataset.diagnosis.value_counts()

# view of descriptive statistics
dataset[numeric_columns].describe()

# create a pairplot
sns.pairplot(dataset[numeric_columns])
#plt.show()


print("*****************************************")
#Data Preparation
print("missing values:")
missing_values=dataset.isnull().sum()
print(missing_values)

# fill missing values with mode
dataset['no_of_vessels'].fillna(dataset['no_of_vessels'].mode()[0], inplace=True)
dataset['thal'].fillna(dataset['thal'].mode()[0], inplace=True)

# extract the target variable
X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
print(X.shape)
print(y.shape)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2606)
print ("train_set_x shape: " + str(X_train.shape))
print ("train_set_y shape: " + str(y_train.shape))
print ("test_set_x shape: " + str(X_test.shape))
print ("test_set_y shape: " + str(y_test.shape))


############### scale feature matrices
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#pca = PCA(n_components = None)
pca = PCA(n_components = 5)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

print("*****************************************")

# create a correlation heatmap
sns.heatmap(dataset[numeric_columns].corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
#plt.show()

def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):
    
    # load model
    model = classifier(**kwargs)
    
    # train model
    model.fit(X_train,y_train)
    
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"Train accuracy: {fit_accuracy:0.2%}")
    print(f"Test accuracy: {test_accuracy:0.2%}")
    
    return model

# KNN
model = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier)
knn_score = model.score(X_test, y_test)
print("knn initial score: "+str(knn_score))
knn_scoreList = []
# Search optimal 'n_neighbours' parameter
print("search optimal of KNN")
for i in range(1,10):
    print("n_neigbors = "+str(i))
    model = train_model(X_train, y_train, X_test, y_test, KNeighborsClassifier, n_neighbors=i)
    knn_scoreList.append(model.score(X_test, y_test))


print("knn max score: "+str(knn_score))
# Decision Tree
model = train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, random_state=2606)
# plot feature importances
## ValueError: Length of passed values is 2, index implies 13.
## pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh()
dt_score = model.score(X_test, y_test)
print("decision tree initial score: "+str(dt_score))


# Logistic Regression
model = train_model(X_train, y_train, X_test, y_test, LogisticRegression)
lg_score=model.score(X_test, y_test)
#Gaussian Naive Bayes
model = train_model(X_train, y_train, X_test, y_test, GaussianNB)
nb_score=model.score(X_test, y_test)

# Support Vector Machines
model = train_model(X_train, y_train, X_test, y_test, SVC)
svm_score=model.score(X_test, y_test)
# tuned SVM
model = train_model(X_train, y_train, X_test, y_test, SVC, C=0.05, kernel='linear')
lineer_svm__score=model.score(X_test, y_test)

print("*****************************************")
methods_accuracy = {
    "KNN" : knn_score,
    "Decision Tree" : dt_score,
    "Logistic Regression":lg_score,
    "Naive Bayes" : nb_score,
    "SVM" : svm_score,
    "Lineer SVM" : lineer_svm__score,
    
}
methods = ["KNN", "Decision Tree","Logistic Regression","Naive Bayes","SVM", "Lineer SVM" ]
accuracy = [knn_score, dt_score, lg_score, nb_score, svm_score, lineer_svm__score]

sns.set()
plt.figure(figsize=(14,6))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=methods, y=accuracy, palette="deep")


for line in range(len(methods)):
     plt.text(line-0.15, # x
              0.70, # y
             "{:.2f}%".format(accuracy[line]*100), 
             horizontalalignment='left',
              size='large',
             color="white",
             )

        
plt.savefig('compare results PCA.png',transparent=True)

        
plt.show()

