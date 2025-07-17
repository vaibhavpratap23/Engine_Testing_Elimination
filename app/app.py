from flask import Flask, render_template, request, session, redirect, url_for
from datetime import datetime
import joblib
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'your-secret-key-123'  # Change this for production

# Load models and scalers
fuel_model = joblib.load(os.path.join('models/fuel_efficiency_model.pkl'))
failure_model = joblib.load(os.path.join('models/failure_risk_model.pkl'))

fuel_scaler = joblib.load(os.path.join('scalers/fuel_scaler.pkl'))
failure_scaler = joblib.load(os.path.join('scalers/failure_scaler.pkl'))

# Initialize session storage if not exists
@app.before_request
def before_request():
    if 'test_history' not in session:
        session['test_history'] = []

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    failure_risk_result = None

    if request.method == 'POST':
        try:
            # Collect data from form
            engine_name = request.form.get('engine_name', 'Unnamed Engine')
            inputs = [
                float(request.form['engine_rpm']),
                float(request.form['fuel_flow_rate']),
                float(request.form['intake_air_temp']),
                float(request.form['coolant_temp']),
                float(request.form['manifold_pressure']),
                float(request.form['ambient_temp']),
                float(request.form['engine_load']),
                float(request.form['exhaust_temp']),
            ]
            
            inputs_np = np.array([inputs])

            # Predict fuel efficiency (regression)
            scaled_input_fuel = fuel_scaler.transform(inputs_np)
            fuel_efficiency = fuel_model.predict(scaled_input_fuel)[0]

            # Predict failure risk (classification)
            scaled_input_failure = failure_scaler.transform(inputs_np)
            failure_risk = failure_model.predict(scaled_input_failure)[0]

            failure_risk_result = "⚠️ RISK" if failure_risk == 1 else "✅ NO RISK"
            prediction_result = f"{fuel_efficiency:.2f} km/l"

            # Store test results
            test_data = {
                'id': len(session['test_history']) + 1,
                'engine_name': engine_name,
                'test_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'fuel_efficiency': f"{fuel_efficiency:.2f}",
                'risk_status': 'High' if failure_risk == 1 else 'Low',
                'details': {
                    'engine_rpm': request.form['engine_rpm'],
                    'fuel_flow_rate': request.form['fuel_flow_rate'],
                    'intake_air_temp': request.form['intake_air_temp'],
                    'coolant_temp': request.form['coolant_temp'],
                    'manifold_pressure': request.form['manifold_pressure'],
                    'ambient_temp': request.form['ambient_temp'],
                    'engine_load': request.form['engine_load'],
                    'exhaust_temp': request.form['exhaust_temp']
                }
            }
            
            session['test_history'] = [test_data] + session['test_history']
            session.modified = True

        except Exception as e:
            prediction_result = f"Error: {str(e)}"
            failure_risk_result = "Prediction Failed"

    return render_template('index.html',
                         prediction_result=prediction_result,
                         failure_risk_result=failure_risk_result)

@app.route('/analytics')
def analytics():
    test_history = session.get('test_history', [])
    
    # Prepare data for chart (last 10 tests)
    recent_tests = test_history[:10]
    test_dates = [test['test_date'][:10] for test in recent_tests][::-1]
    efficiency_data = [float(test['fuel_efficiency'].split()[0]) for test in recent_tests][::-1]
    
    return render_template('analytics.html',
                         test_history=test_history,
                         test_dates=test_dates,
                         efficiency_data=efficiency_data)

@app.route('/test-details/<int:test_id>')
def test_details(test_id):
    test_history = session.get('test_history', [])
    test = next((t for t in test_history if t['id'] == test_id), None)
    
    if test:
        return render_template('test_details.html', test=test)
    else:
        return redirect(url_for('analytics'))
    
@app.route('/project-report')
def project_report():
    return render_template('project_report.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/list-routes')
def list_routes():
    return '<br>'.join([str(rule) for rule in app.url_map.iter_rules()])

if __name__ == '__main__':
    app.run(debug=True)