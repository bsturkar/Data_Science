from flask import Flask,render_template,request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()




# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home1.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data=[message]
        corpus = []
        review = re.sub('[^a-zA-Z]', ' ',message)
        review = review.lower()
        review = review.split()
    
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        #below command will join the words and will form a sentences 
        review = ' '.join(review)
        corpus.append(review)
        vect=cv.transform(corpus).toarray()
        my_prediction=clf.predict(vect)
    return render_template('result1.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)