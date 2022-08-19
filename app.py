import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle
app = Flask(__name__)



#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
#------------------------------About us-------------------------------------------
@app.route('/aboutus')
def aboutusnew():
    return render_template('aboutus.html')
@app.route('/Svm')
def svm():
    return render_template('Svm.html')
@app.route('/decision')
def decisiontree(): 
    return render_template('decision.html')
@app.route('/knn')
def knn():
        return render_template('knn.html')
@app.route('/Rf')
def randomforest():
    return render_template('Rf.html')
@app.route('/nvb')
def naivebayes():
    return render_template('nvb.html')
@app.route('/inter')
def inter():
    return render_template('inter.html')

@app.route('/code')

def QR():
    return render_template('QR.html')

@app.route('/predict',methods=['GET'])
def predict():
    
  model=pickle.load(open('anmol2_Major_decision.pkl','rb'))
    
  text1 = request.args.get('text')
    
  dataset = pd.read_csv('spam.csv', encoding='latin-1')

  import re
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  corpus = []
  for i in range(0, dataset.shape[0]):
      review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[:,1][i])
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      review = ' '.join(review)
      corpus.append(review) 
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  X = cv.fit_transform(corpus).toarray()
  review = review.split()
  
    
  X = cv.transform(review).toarray()
  input_pred = model.predict(X)
  input_pred = input_pred.astype(int)
  
    
  if input_pred[0]==1:
      result= "Spam"
  else:
      result="Not Spam"
    
            
  return render_template('predict.html', prediction_text='Model  has predicted  : {}'.format(result))
    
@app.route('/predict1',methods=['GET'])
def predict1():
   
  model=pickle.load(open('anmol2_major_project_knn.pkl','rb'))
  text1 = request.args.get('text')
    
  dataset = pd.read_csv('spam.csv', encoding='latin-1')

  import re
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  corpus = []
  for i in range(0, dataset.shape[0]):
      review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[:,1][i])
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      review = ' '.join(review)
      corpus.append(review) 
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  X = cv.fit_transform(corpus).toarray()
  review = review.split()
  
    
  X = cv.transform(review).toarray()
  input_pred = model.predict(X)
  input_pred = input_pred.astype(int)
  
    
  if input_pred[0]==1:
      result= "Spam"
  else:
      result="Not Spam"
    
 
            
  return render_template('predict.html', prediction_text='Model  has predicted  : {}'.format(result))


@app.route('/predict2',methods=['GET'])
def predict2():
   
  model=pickle.load(open('anmol2_major_project_svm.pkl','rb'))

  text1 = request.args.get('text')
    
  dataset = pd.read_csv('spam.csv', encoding='latin-1')

  import re
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  corpus = []
  for i in range(0, dataset.shape[0]):
      review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[:,1][i])
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      review = ' '.join(review)
      corpus.append(review) 
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  X = cv.fit_transform(corpus).toarray()
  review = review.split()
  
    
  X = cv.transform(review).toarray()
  input_pred = model.predict(X)
  input_pred = input_pred.astype(int)
  
    
  if input_pred[0]==1:
      result= "Spam"
  else:
      result="Not Spam"
    
    
 
            
  return render_template('predict.html', prediction_text='Model  has predicted  : {}'.format(result))
@app.route('/predict3',methods=['GET'])
def predict3():
   
  model=pickle.load(open('anmol2_major_project_random.pkl','rb'))

  text1 = request.args.get('text')
    
  dataset = pd.read_csv('spam.csv', encoding='latin-1')

  import re
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  corpus = []
  for i in range(0, dataset.shape[0]):
      review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[:,1][i])
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      review = ' '.join(review)
      corpus.append(review) 
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  X = cv.fit_transform(corpus).toarray()
  review = review.split()
  
    
  X = cv.transform(review).toarray()
  input_pred = model.predict(X)
  input_pred = input_pred.astype(int)
  
    
  if input_pred[0]==1:
      result= "Spam"
  else:
      result="Not Spam"
    
 
            
  return render_template('predict.html', prediction_text='Model  has predicted  : {}'.format(result))

@app.route('/predict4',methods=['GET'])
def predict4():
   
  model=pickle.load(open('anmol2_major_naive.pkl','rb'))

  text1 = request.args.get('text')
    
  dataset = pd.read_csv('spam.csv', encoding='latin-1')

  import re
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  from nltk.stem.porter import PorterStemmer
  corpus = []
  for i in range(0, dataset.shape[0]):
      review = re.sub('[^a-zA-Z]', ' ', dataset.iloc[:,1][i])
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      review = ' '.join(review)
      corpus.append(review) 
  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer(max_features = 1500)
  X = cv.fit_transform(corpus).toarray()
  review = review.split()
  
    
  X = cv.transform(review).toarray()
  input_pred = model.predict(X)
  input_pred = input_pred.astype(int)
  
    
  if input_pred[0]==1:
      result= "Spam"
  else:
      result="Not Spam"
    
 
            
  return render_template('predict.html', prediction_text='Model  has predicted  : {}'.format(result))



if __name__ == "__main__":
  app.run()
