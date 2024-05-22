import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

file = pd.read_csv("emails.csv")

#defining input and output
x = file['text']
y = file['spam']

#splitting data into test and train
train, test = train_test_split(file, test_size = 0.2)#split data into test and train
x_train = train['text']
y_train = train['spam']
x_test = test['text']
y_test = test['spam']

cv = CountVectorizer()
#IDing each word and counting number of occurrences of said word
features = cv.fit_transform(x_train)

#building model
model = DecisionTreeClassifier()
model.fit(features, y_train)

#testing
features_test = cv.transform(x_test)
predictions = model.predict(features_test)

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, predictions))
print("\n\nReport:\n")
print(classification_report(y_test, predictions))
