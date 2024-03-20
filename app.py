from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

scaler = pickle.load(open("standardScalar.pkl", "rb"))
model = pickle.load(open("modelForPrediction.pkl", "rb"))

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)
    new_data = scaler.transform(data)
    prediction = model.predict(new_data)
    return prediction[0]

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        diabetes_pedigree_function = float(request.form["diabetes_pedigree_function"])
        age = float(request.form["age"])

        prediction = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)

        if prediction == 1:
            result = "Diabetic"
        else:
            result = "Non-Diabetic"

        return render_template("index.html", prediction=result)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
