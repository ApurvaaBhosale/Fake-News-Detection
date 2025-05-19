import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

fake= pd.read_csv(r"C:\Project1\data\Fake.csv")
true= pd.read_csv(r"C:\Project1\data\True.csv")

#Assinging labels
fake['label'] = 0
true['label'] = 1
#Merging
data = pd.concat([fake, true])
data = data.sample(frac= 1).reset_index(drop=True)# Shuffling the data

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    return text

data['text']= data['text'].apply(clean_text)

data = data[['text','label']]#keeping only the columns of text and label

vector = TfidfVectorizer(stop_words='english')
x=vector.fit_transform(data['text'])
y=data['label']

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=42)

model=PassiveAggressiveClassifier(max_iter=500)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test, y_pred)
cm=confusion_matrix(y_test, y_pred)
print("Accuracy:",accuracy)
print(cm)

import os
import joblib

# Get current file's directory (safe even if script is run from elsewhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'app', 'model')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Save model and vectorizer
joblib.dump(model, os.path.join(MODEL_DIR, 'fake_news_model.pkl'))
joblib.dump(vector, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))


