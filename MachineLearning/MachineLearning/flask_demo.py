from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import pandas as pd
import pickle
from file import testflask

my_model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')
@app.route('/predict',methods=['POST'])

def predict():
    data = request.form
    hair= float(data['hair'])
    feathers=float(data['feathers'])
    eggs=float(data['eggs'])
    milk=float(data['milk'])
    airbone=float(data['airbone'])
    aquatic=float(data['aquatic'])
    predator=float(data['predator'])
    toothed=float(data['toothed'])
    backbone=float(data['backbone'])
    breathes=float(data['breathes'])
    venomous=float(data['venemous'])
    fins=float(data['fins'])
    legs=float(data['legs'])
    tail=float(data['tail'])
    domestic=float(data['domestic'])
    catsize=float(data['catsize'])
    
    
    predict_request = [[hair,feathers,eggs,milk,airbone,aquatic,predator,toothed,backbone,breathes,
                        venomous,fins,legs,tail,domestic,catsize]]
    predict_request=np.array(predict_request)
    data=pd.DataFrame(predict_request, columns = ['hair','feathers','eggs','milk',
                                                   'airbone','aquatic','predator','toothed','backbone',
                                                  'breathes','venomous','fins','legs','tail','domestic','catsize'])
    #predictions= my_model.predict(data)
    acc=int(testflask(data, my_model))
    return render_template('results.html', prediction=acc)

if __name__ == '__main__':
    app.run(debug=True)