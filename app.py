from flask import Flask, render_template,request
import pickle

app=Flask(__name__)

cv=pickle.load(open("models/cv.pkl","rb"))
clf=pickle.load(open("models/clf.pkl","rb"))

@app.route("/")
def home():
    # text=""
    # if request.method=="POST":
    #     text = request.form.get('email-content')
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    email=request.form.get('content')
    tokenized_email=cv.transform([email])
    predictions=clf.predict(tokenized_email)
    predictions=1 if predictions==1 else -1
    return render_template("index.html",predictions=predictions,email_text=email)

if __name__=="__main__":
    app.run(debug=True)