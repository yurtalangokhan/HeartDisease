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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve


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



##Analyze features and explore the data

# Distribution of those who are sick and those who are not
f, ax = plt.subplots(1,2,figsize=(14,6))
dataset['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('diagnosis')
ax[0].set_ylabel('')
sns.countplot('diagnosis', data=dataset, ax=ax[1])
plt.savefig('Disease.png',transparent=True)
plt.show()

#Distribution of sex
plt.figure(figsize=(12,7))
sns.set()
sns.countplot(x='sex', data=dataset, palette=color)
plt.xlabel("Gender (0 = female, 1= male)")
plt.ylabel("Count of person")
plt.savefig('gender.png',transparent=True)

plt.show()

print("*****************************************")
countFemale = len(dataset[dataset.sex == 0]) # Count of Female
countMale = len(dataset[dataset.sex == 1]) #  Count of Male
print("Percentage of female: {:.2f}%".format((countFemale / (len(dataset.sex))*100)))
print("Percentage of male: {:.2f}%".format((countMale / (len(dataset.sex))*100)))

pd.crosstab(dataset.sex,dataset.diagnosis).plot(kind="bar",figsize=(15,6),color=color)
plt.title('Frequency of Heart Disease by Gender')
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["not sick.", "sick"])
plt.ylabel('frequency')
plt.savefig('diseaseAccordingToGender.png',transparent=True)
plt.show()

#Distribution of Age per disease
print ("Mean age per disease type")
val=dataset.groupby(["diagnosis", ])["age"].mean()
print(val)

pd.crosstab(dataset.age,dataset.diagnosis).plot(kind="bar",figsize=(20,9),color=color)
plt.title('Heart Disease by Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png',transparent=True)
plt.show()


# Disease Distribution Between Maximum Heart Rate and Age

plt.figure(figsize=(12,8))
plt.scatter(x=dataset.age[dataset.diagnosis==1], y=dataset.thal[(dataset.diagnosis==1)], c="red")
plt.scatter(x=dataset.age[dataset.diagnosis==0], y=dataset.thal[(dataset.diagnosis==0)])
plt.legend(["Sick", "Not Sick"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.savefig('maximumHeartRate.png',transparent=True)
plt.show()

#Disease Frequency by Slope Variable
color = ["#58a3bc","#666666"]
pd.crosstab(dataset.slope,dataset.diagnosis).plot(kind="bar",figsize=(15,6),color=color)
plt.title('Disease Frequency by Slope Variable')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.savefig('slope.png',transparent=True)
plt.show()

pd.crosstab(dataset.fasting_blood_sugar,dataset.diagnosis).plot(kind="bar",figsize=(15,6),color=color)
plt.title('Frequency of Heart Disease by Fasting Blood Sugar')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Not Sick", "Sick"])
plt.ylabel('Frequency of Sick or Not Sick')
plt.savefig('hunger.png',transparent=True)
plt.show()

# create a barplot
sns.barplot(x="fasting_blood_sugar", y="diagnosis", data=dataset)

pd.crosstab(dataset.chest_pain ,dataset.diagnosis).plot(kind="bar",figsize=(15,6),color=color)
plt.title('Frequency of Halp Disease by Type of Chest Pain')
plt.xlabel('Chest Pain Type (4 Values)')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Sick or Not Sick')
plt.savefig('chest.png',transparent=True)
plt.show()

# create a correlation heatmap
sns.heatmap(dataset[numeric_columns].corr(),annot=True, cmap='terrain', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.savefig('correlationHeatMap.png',transparent=True)
plt.show()

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




def train_model(X_train, y_train, X_test, y_test, classifier, **kwargs):

    print("------------------ Train & Test Results----------------")
    # load model
    model = classifier(**kwargs)
    
    # train model
    model.fit(X_train,y_train)
    
    # check accuracy and print out the results
    fit_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Train accuracy: {fit_accuracy:0.2%}")
    print(f"Test accuracy: {test_accuracy:0.2%}")
    
    print("---------------------")
    
    return model
    
def result_model(X_train, y_train, X_test, y_test,method_name):

    print("################## " + method_name + " is training ########################")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) 
    precision = precision_score(y_test, y_pred) 
    recall = recall_score(y_test, y_pred)
    
    print(f"accuracy: {accuracy:0.2%}")
    print(f"precision: {precision:0.2%}")
    print(f"recall: {recall:0.2%}")
    print("######################")
    
    return y_pred

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

knn_score = max(knn_scoreList)
print("knn max score: "+str(knn_score))
y_pred_knn = result_model(X_train, y_train, X_test, y_test,"KNN")

# Decision Tree
model = train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, random_state=2606)
# plot feature importances
pd.Series(model.feature_importances_,X.columns).sort_values(ascending=True).plot.barh()
dt_score = model.score(X_test, y_test)
print("decision tree initial score: "+str(dt_score))

dt_scoreList = []
# Search optimal 'max_depth' parameter
print("search optimal of Decision Tree")
for i in range(1,8):
    print("max_depth = "+str(i))
    model= train_model(X_train, y_train, X_test, y_test, DecisionTreeClassifier, max_depth=i, random_state=2606)
    dt_scoreList.append(model.score(X_test, y_test))

dt_score = max(dt_scoreList)
y_pred_dt = result_model(X_train, y_train, X_test, y_test,"Decision Tree")
# Logistic Regression
model = train_model(X_train, y_train, X_test, y_test, LogisticRegression)
lg_score=model.score(X_test, y_test)
y_pred_lg = result_model(X_train, y_train, X_test, y_test,"Logistic Regression")
#Gaussian Naive Bayes
model = train_model(X_train, y_train, X_test, y_test, GaussianNB)
nb_score=model.score(X_test, y_test)
y_pred_nb = result_model(X_train, y_train, X_test, y_test,"Gaussian Naive Bayes")

# Support Vector Machines
model = train_model(X_train, y_train, X_test, y_test, SVC)
svm_score=model.score(X_test, y_test)
y_pred_svm= result_model(X_train, y_train, X_test, y_test,"SVM")
# tuned SVM
model = train_model(X_train, y_train, X_test, y_test, SVC, C=0.05, kernel='linear')
lineer_svm__score=model.score(X_test, y_test)
y_pred_linear_svm= result_model(X_train, y_train, X_test, y_test,"SVM Linear")



print("roc_curve *****************************************")
plt.figure(figsize = (10,10))
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn)
plt.plot(fpr, tpr, color='orange', label='KNN')

fpr1, tpr1, thresholds = roc_curve(y_test, y_pred_dt)
plt.plot(fpr1, tpr1, color='orange', label='Decision Tree')

fpr2, tpr2, thresholds = roc_curve(y_test, y_pred_lg)
plt.plot(fpr2, tpr2, color='purple', label='Logistic Regression')

fpr3, tpr3, thresholds = roc_curve(y_test, y_pred_nb)
plt.plot(fpr3, tpr3, color='pink', label='Gaussian Naive Bayes')

fpr4, tpr4, thresholds = roc_curve(y_test, y_pred_svm)
plt.plot(fpr4, tpr4, color='green', label='SVM')

fpr5, tpr5, thresholds = roc_curve(y_test, y_pred_linear_svm)
plt.plot(fpr5, tpr5, color='red', label='Linear SVM')

plt.legend()
plt.savefig('resultsRocCurve.png',transparent=True)
plt.show()



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

        
plt.savefig('compare results.png',transparent=True)

        
plt.show()

