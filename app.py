import pickle

from flask import Flask, render_template, request
from pandas import np

app = Flask(__name__)

with open(r'C:\Users\KAMRAN\PycharmProjects\fakeOrNot\pickle_model', "rb") as f:
    pm = pickle.load(f)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/home')
def home():
    return render_template("Index.html")


@app.route('/about')
def about():
    return render_template("About.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = pm.predict(final_features)

    print(prediction)

    result=prediction[0]
    print(result)
    #output = round(prediction[0], 2)

    if result == 1:
        return render_template('index.html',
                               dpredicts='Account with these attribute values is Fake.' .format(result))
    else:
        return render_template('index.html',
                               predicts='Account with these attribute values is not Fake.'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
