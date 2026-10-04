from flask import Flask , render_template , request
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from feature.feature import feature_engineering
import joblib

# from sklearn.utils.class_weight import compute_class_weight
model = joblib.load("model.h5")
train_data = pd.read_csv('train_data.csv')
train_data = train_data.drop(['UDI','Product ID', 'Machine failure'],axis=1)

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/predict',methods=["POST"])
def predict():
    if request.method == "POST":
        data = {
        'Air temperature [K]': float(request.form['Air_temperature_K']),
        'Process temperature [K]': float(request.form['Process_temperature_K']),
        'Rotational speed [rpm]': float(request.form['Rotational_speed_rpm']),
        'Torque [Nm]': float(request.form['Torque_Nm']),
        'Tool wear [min]': float(request.form['Tool_wear_min']),
        'Type': request.form['type'],
        'TWF': int(request.form['TWF']),
        'HDF': int(request.form['HDF']),
        'PWF': int(request.form['PWF']),
        'OSF': int(request.form['OSF']),
        'RNF': int(request.form['RNF'])
            }

    input_data = pd.DataFrame([data])
    print(input_data.shape)

    Fe = feature_engineering()
    train_data_preprocessed , input_data = Fe.preprocessing(input_data, train_data=train_data)
    # user_processed = Fe.preprocessing(input_data)

    print(input_data.shape)
    y_proba = model.predict(input_data).ravel()
    prediction = ("Machine Falure" if y_proba >= 0.5 else "No Machine Falure")

    return render_template('index.html',prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)