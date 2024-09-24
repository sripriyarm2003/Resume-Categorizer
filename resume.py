import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
plt.style.use('ggplot')

df = pd.read_csv('resume.csv')

print(df.head())      #initial 5

print(df.sample(5))         #random 5
print(df.shape)              #rows,col
print(df['Category'].unique())       #unique list
print(df['Category'].value_counts())      #counting

print(plt.figure(figsize=(15,5)))
df['Category'].value_counts().plot(kind="bar")      #bar graph
plt.show()

counts = df['Category'].value_counts()
labels = df['Category'].unique()
plt.figure(figsize=(15,10))
plt.pie(counts, labels = labels, autopct = '%1.1f%%', colors=plt.cm.Blues(np.linspace(0,1,3)))  #pie chart 
plt.show()

print(df['Resume'][0])

def clean(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    clean_text = url_pattern.sub('', text)
    clean_text = email_pattern.sub('', clean_text)
    
    clean_text = re.sub('[^\w\s]', '', clean_text)
    stop_words = set(stopwords.words('english'))
    clean_text  = ' '.join(word for word in clean_text.split() if word.lower() not in stop_words)
    
    return clean_text

s = 'Hello, $%& I found this # article at https://www.chando.com ! You can contact me at chando@example.com'
print(clean(s))

df['Resume'] = df['Resume'].apply(lambda x:clean(x))
print(df['Resume'][0])

le = LabelEncoder()
le.fit(df[['Category']])
df['Category'] = le.transform(df['Category'])
print(df['Category'].unique())

tfidf = TfidfVectorizer()
tfidf.fit(df['Resume'])
resume = tfidf.transform(df['Resume'])
X_train , X_test, y_train , y_test = train_test_split(resume, df['Category'], test_size = 0.2, random_state = 42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuarcy of the KNN Classifier on test data-> : {accuracy}")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

models = {
    'KNeighborsClassifier':KNeighborsClassifier(),
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(),
    'MultinomialNB': MultinomialNB(),
    'OneVsRestClassifier': OneVsRestClassifier(KNeighborsClassifier()) 
}
accuracy_scores = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    accuracy_scores[model_name] = accuracy
    print(f"Accuarcy of {model_name} on test data: {accuracy}")

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
myresume = """Education: Bachelor of Engineering in Computer Science, Vidyavardhaka College of Engineering, Mysuru (2021-2025), CGPA: 8.98. Skills: C Programming, Java, DBMS (SQL), Web Development (HTML, CSS), Google Cloud Platform. Experience: 1. Weather App - Developed a responsive web application using OpenWeather API, HTML, CSS, and JavaScript. 2. Resume Categorizer - Python-based application using TF-IDF and machine learning (sklearn, NLTK, TF-IDF, Pandas, Pickle) to classify resumes. 3. Personal Portfolio - Developed a responsive portfolio website using HTML, CSS, JavaScript, and GitHub. Certifications: Google Cloud Associate Cloud Engineer, Programming in C - Infosys Springboard, Basics of Python - Infosys Springboard. Volunteering: Code Club Member (VVCE), YFS (Youth for Seva), NSS Member (National Service Scheme)."""
print(df.head())
cleaned_resume = clean(myresume)
input_features = tfidf.transform([cleaned_resume])
prediction_id = model.predict(input_features)[0]
category_map = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

category_name = category_map.get(prediction_id, "Unknown")
print("Predicted Category is-> ", category_name)

import pickle
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))