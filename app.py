from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

with open('voting_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def recommend_crop(pH, phosphorus, potassium, urea, temperature):
    try:
        pH = float(pH)
        phosphorus = float(phosphorus)
        potassium = float(potassium)
        urea = float(urea)
        temperature = float(temperature)
        

        pH_mean, pH_std = 6.311596, 0.424663 
        phosphorus_mean, phosphorus_std = 23.479479, 8.756160 
        potassium_mean, potassium_std = 146.067966, 47.294006 
        urea_mean, urea_std = 52.474345, 20.965486  
        temperature_mean, temperature_std = 72.532048,8.950912  
        
        pH_normalized = (pH - pH_mean) / pH_std
        phosphorus_normalized = (phosphorus - phosphorus_mean) / phosphorus_std
        potassium_normalized = (potassium - potassium_mean) / potassium_std
        urea_normalized = (urea - urea_mean) / urea_std
        temperature_normalized = (temperature - temperature_mean) / temperature_std
        
        input_data = pd.DataFrame({
            'pH': [pH_normalized],
            'Phosphorus': [phosphorus_normalized],
            'Potassium': [potassium_normalized],
            'Urea': [urea_normalized],
            'Temperature': [temperature_normalized]
        })
        prediction = model.predict(input_data)[0]
        return prediction
    except ValueError:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend_crop', methods=['POST'])
def recommend_crop_route():
    pH = request.form['pH']
    phosphorus = request.form['phosphorus']
    potassium = request.form['potassium']
    urea = request.form['urea']
    temperature = request.form['temperature']

    try:
        recommended_crop = recommend_crop(pH, phosphorus, potassium, urea, temperature)
        if recommended_crop is not None:
            return jsonify({'recommended_crop': recommended_crop})
        else:
            return jsonify({'error': 'Please enter valid numerical values for all input parameters.'}), 400
    except ValueError:
        return jsonify({'error': 'Please enter valid numerical values for all input parameters.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
