from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and preprocessing objects
model = joblib.load('model/titanic_survival_model.pkl')
scaler = joblib.load('model/scaler.pkl')
embarked_encoder = joblib.load('model/embarked_encoder.pkl')
config = joblib.load('model/model_config.pkl')

# Get embarked options
embarked_options = list(embarked_encoder.classes_)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', embarked_options=embarked_options)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from form
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']
        
        # Encode embarked
        embarked_encoded = embarked_encoder.transform([embarked])[0]
        
        # Create feature array in correct order: Pclass, Sex, Age, Fare, Embarked_Encoded
        features = np.array([[pclass, sex, age, fare, embarked_encoded]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare result
        if prediction == 1:
            result = "✅ SURVIVED"
            confidence = probability[1] * 100
            color = "success"
        else:
            result = "❌ DID NOT SURVIVE"
            confidence = probability[0] * 100
            color = "danger"
        
        return render_template('index.html',
                             prediction_text=result,
                             confidence=f"{confidence:.1f}%",
                             color=color,
                             embarked_options=embarked_options)
    
    except Exception as e:
        return render_template('index.html',
                             prediction_text=f'Error: {str(e)}',
                             color='warning',
                             embarked_options=embarked_options)

if __name__ == '__main__':
    app.run(debug=True)