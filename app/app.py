from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
import joblib
import numpy as np
import os
import json
import traceback

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your-secret-key-keep-this-safe'

# Initialize model variables
fuel_model = None
failure_model = None
fuel_scaler = None
failure_scaler = None
models_loaded = False

def load_models_and_scalers():
    global fuel_model, failure_model, fuel_scaler, failure_scaler, models_loaded
    
    try:
        # Get the directory where app.py is located
        APP_DIR = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the Engine_testing folder
        PROJECT_ROOT = os.path.dirname(APP_DIR)
        
        # Define paths to models and scalers
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        scalers_dir = os.path.join(PROJECT_ROOT, 'scalers')
        
        # Print debug information
        print("\n=== Loading Models and Scalers ===")
        print(f"Project Root: {PROJECT_ROOT}")
        print(f"Models Directory: {models_dir}")
        print(f"Scalers Directory: {scalers_dir}\n")

        # Load each file with verification
        def load_file(path, file_type):
            if os.path.exists(path):
                print(f"✓ {file_type} found at: {path}")
                return joblib.load(path)
            else:
                print(f"× {file_type} not found at: {path}")
                return None

        fuel_model = load_file(os.path.join(models_dir, 'fuel_efficiency_model.pkl'), "Fuel Efficiency Model")
        failure_model = load_file(os.path.join(models_dir, 'failure_risk_model.pkl'), "Failure Risk Model")
        fuel_scaler = load_file(os.path.join(scalers_dir, 'fuel_scaler.pkl'), "Fuel Scaler")
        failure_scaler = load_file(os.path.join(scalers_dir, 'failure_scaler.pkl'), "Failure Scaler")

        models_loaded = all([fuel_model, failure_model, fuel_scaler, failure_scaler])
        
        if models_loaded:
            print("\n✓ All models and scalers loaded successfully!")
        else:
            print("\n× Some models/scalers failed to load. Running in demo mode.")
            
    except Exception as e:
        print(f"\nError loading models: {str(e)}")
        traceback.print_exc()
        models_loaded = False

# Load models when starting the app
load_models_and_scalers()

def get_data_file_path():
    """Get the path to the test data JSON file in the data folder"""
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(APP_DIR)
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, 'test_data.json')

def load_test_history():
    """Load the test history from JSON file"""
    try:
        data_file = get_data_file_path()
        if not os.path.exists(data_file):
            return []
        with open(data_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading test history: {str(e)}")
        return []

def save_test_history(data):
    """Save test history to JSON file"""
    try:
        with open(get_data_file_path(), 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving test history: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if not models_loaded:
                return render_template('index.html', 
                                     error="Models not loaded - running in demo mode with sample data")

            # Get form data
            engine_data = {
                'rpm': float(request.form['rpm']),
                'temperature': float(request.form['temperature']),
                'pressure': float(request.form['pressure']),
                'vibration': float(request.form['vibration']),
                'oil_level': float(request.form['oil_level']),
                'hours_used': float(request.form['hours_used'])
            }
            
            # Prepare input features
            input_features = np.array([
                engine_data['rpm'],
                engine_data['temperature'],
                engine_data['pressure'],
                engine_data['vibration'],
                engine_data['oil_level'],
                engine_data['hours_used']
            ]).reshape(1, -1)
            
            # Make predictions
            scaled_fuel_input = fuel_scaler.transform(input_features)
            fuel_efficiency = round(float(fuel_model.predict(scaled_fuel_input)[0]), 2)
            
            scaled_failure_input = failure_scaler.transform(input_features)
            failure_prob = failure_model.predict_proba(scaled_failure_input)
            failure_risk = round(float(failure_prob[0][1]) * 100, 2)  # Convert to percentage
            
            # Save test results
            test_history = load_test_history()
            test_id = len(test_history) + 1
            test_result = {
                'id': test_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'input_data': engine_data,
                'results': {
                    'fuel_efficiency': fuel_efficiency,
                    'failure_risk': failure_risk
                }
            }
            test_history.append(test_result)
            save_test_history(test_history)
            
            return render_template('index.html', 
                                success=True,
                                fuel_efficiency=fuel_efficiency,
                                failure_risk=failure_risk,
                                test_id=test_id)
            
        except Exception as e:
            traceback.print_exc()
            return render_template('index.html', 
                                error=f"Error processing request: {str(e)}")
    
    # GET request - just show the form
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    test_history = load_test_history()
    return render_template('analytics.html', 
                         test_history=test_history,
                         success=request.args.get('success'))

@app.route('/test-details/<int:test_id>')
def test_details(test_id):
    test_history = load_test_history()
    test = next((t for t in test_history if t['id'] == test_id), None)
    if test:
        return render_template('test_details.html', test=test)
    return redirect(url_for('analytics'))

@app.route('/project-report')
def project_report():
    return render_template('project_report.html')

@app.route('/support')
def support():
    return render_template('support.html')

if __name__ == '__main__':
    # Print startup information
    print("\n=== Starting Engine Testing Application ===")
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    print(f"Data file location: {get_data_file_path()}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)