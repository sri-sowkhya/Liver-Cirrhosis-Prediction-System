from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("rf_acc_68.pkl", "rb"))
scaler = pickle.load(open("normalizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html", form_values={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form.to_dict()
        print("Form received:", form)

        gender = 1 if form.get('Gender', '').lower() == 'male' else 0
        obesity = 1 if form.get('Obesity', '').lower() == 'yes' else 0
        diabetes = 1 if form.get('Diabetes Result', '').lower() == 'yes' else 0

        data = [
            float(form['Age']),
            gender,
            float(form['Duration of alcohol consumption(years)']),
            float(form['Quantity of alcohol consumption (quarters/day)']),
            diabetes,
            obesity,
            float(form['TCH']),
            float(form['LDL']),
            float(form['Hemoglobin  (g/dl)']),
            float(form['SGPT/ALT (U/L)'])
        ]

        columns = ["Age", "Gender", "AlcoholDuration", "AlcoholQuantity", "Diabetes", "Obesity", "TCH", "LDL", "Hemoglobin", "SGPT"]
        input_df = pd.DataFrame([data], columns=columns)
        normalized_data = scaler.transform(input_df)

        prediction = model.predict(normalized_data)

        result = "Positive for Liver Cirrhosis" if prediction[0] == 1 else "Negative for Liver Cirrhosis"

        return render_template("index.html", prediction_text=result, form_values=form)

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
