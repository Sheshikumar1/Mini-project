from flask import Flask, render_template, request
import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model and encoded values
model = joblib.load(open("model_small.pkl", 'rb'))
encoded = joblib.load('label_values')

# Load the label encoder and fit with all city names
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/output', methods=["POST"])
def output():
    if request.method == 'POST':
        city = request.form["city"].strip()
        pm25 = float(request.form["pm25"])
        pm10 = float(request.form["pm10"])
        no = float(request.form["no"])
        no2 = float(request.form["no2"])
        nox = float(request.form["nox"])
        nh3 = float(request.form["nh3"])
        co = float(request.form["co"])
        so2 = float(request.form["so2"])
        o3 = float(request.form["o3"])
        benzene = float(request.form["benzene"])
        toluene = float(request.form["toluene"])
        xylene = float(request.form["xylene"])
        date = request.form["date"]

        # Ensure the city name is valid
        if city not in label_encoder.classes_:
            return render_template("output.html", y="Invalid City", z="Please enter a valid city name.")

        # Transform city and date fields
        city_encoded = label_encoder.transform([city])[0]
        year = int(date.split('-')[0])
        month = int(date.split('-')[1])

        # Create a DataFrame for the input features
        feature_cols = ['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'Year', 'Month']
        data = pd.DataFrame([[city_encoded, pm25, pm10, no, no2, nox, nh3, co, so2, o3, benzene, toluene, xylene, year, month]], columns=feature_cols)

        # Make prediction
        pred = model.predict(data)
        pred = pred[0]

        # Determine AQI category
        if pred >= 0 and pred < 50:
            res = 'GOOD'
        elif pred >= 50 and pred < 100:
            res = 'SATISFACTORY'
        elif pred >= 100 and pred < 200:
            res = 'MODERATELY POLLUTED'
        elif pred >= 200 and pred < 300:
            res = 'POOR'
        elif pred >= 300 and pred < 400:
            res = 'VERY POOR'
        else:
            res = 'SEVERE'

        return render_template("output.html", y=f"AQI: {str(pred)}", z=res)

if __name__ == "__main__":
    app.run(debug=True)