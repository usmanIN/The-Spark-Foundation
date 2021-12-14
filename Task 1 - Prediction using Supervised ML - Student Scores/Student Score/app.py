from flask import Flask, render_template, request
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
app = Flask(__name__)


file = "../student_scores.csv"
if os.path.exists(file):
    model = open(file,'rb')
else:
    print("File Don't Exists")    
#    exit()


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    result = 0
    if request.method == "POST":
        if request.form['hours'] !="":
            if type(request.form['hours'])!='str':
                hour = float(request.form['hours'])           
                df = pd.read_csv(file)
            
                x = df.Hours # feature value
                y = df.Scores # target value

                x = x.values.reshape(-1,1)
                
                model = LinearRegression() # Prepare the model
                model.fit(x,y) # fit the model

                pred = model.predict([[hour]])

                result = round(pred[0],2)       

    return render_template('index.html',result=result)

if __name__=="__main__":
    app.run(debug=True)      