from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import sqlite3
import os
import joblib
import pandas as pd
import numpy as np
from flask import jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.secret_key = os.urandom(24)  # Secret key for session management

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
try:
    model_botiot = joblib.load('model_botiot.sav')
    model_kdd = joblib.load('model_kdd.sav')
except Exception as e:
    print(f"Error loading models: {e}")
    from sklearn.ensemble import RandomForestClassifier
    model_botiot = RandomForestClassifier()
    model_kdd = RandomForestClassifier()
    print("Warning: Placeholder models are not trained and will raise errors during prediction.")

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Check if user exists in the database
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user and user['password'] == password:  # In real scenarios, hash passwords!
            session['user'] = user['email']
            return redirect(url_for('home'))  # Redirect to home page after login
        else:
            flash("Invalid credentials, please try again.")
            return redirect(url_for('signin'))

    return render_template('signin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Check if the email is already taken
        conn = get_db_connection()
        existing_user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        
        if existing_user:
            flash("Email is already registered, please use a different one.")
            return redirect(url_for('signup'))
        
        # Save new user to the database
        conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', (name, email, password))
        conn.commit()
        conn.close()

        flash("Account created successfully! Please sign in.")
        return redirect(url_for('signin'))  # Redirect to sign-in after successful sign-up

    return render_template('signup.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('signin'))  # Redirect to sign-in if not logged in

    return render_template('home.html')

@app.route('/bot_io', methods=['GET', 'POST'])
def bot_io():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                if file.filename.endswith('.csv'):
                    data = pd.read_csv(filepath)
                elif file.filename.endswith('.xls'):
                    data = pd.read_excel(filepath, engine='xlrd')
                elif file.filename.endswith('.xlsx'):
                    data = pd.read_excel(filepath, engine='openpyxl')
                else:
                    return jsonify({'error': 'Unsupported file format'}), 400

                # Check if model is loaded properly
                if model_botiot is None:
                    print("Error: Model not loaded properly.")
                    return jsonify({'error': 'Model not loaded properly.'}), 500

                # Make predictions
                print(f"Data shape: {data.shape}")
                predictions = model_botiot.predict(data)
                data['Prediction'] = ['Normal' if pred == 0 else 'Attack' for pred in predictions]

                total_records = len(predictions)
                attack_records = np.sum(predictions == 1)
                normal_records = total_records - attack_records
                attack_percentage = (attack_records / total_records) * 100 if total_records > 0 else 0
                normal_percentage = (normal_records / total_records) * 100 if total_records > 0 else 0

                # Convert to Python native types
                attack_records = int(attack_records)
                normal_records = int(normal_records)

                prediction_table = data.to_html(classes='data')
                print(prediction_table)

                return render_template('bot_iot.html',
                                       prediction_table=prediction_table,
                                       total_records=total_records,
                                       attack_records=attack_records,
                                       normal_records=normal_records,
                                       attack_percentage=attack_percentage,
                                       normal_percentage=normal_percentage)

            except Exception as e:
                print(f"Error occurred: {str(e)}")
                # Return HTML response on error, not JSON
                return render_template('bot_iot.html',
                                       prediction_table=None,
                                       total_records=0,
                                       attack_records=0,
                                       normal_records=0,
                                       attack_percentage=0,
                                       normal_percentage=0,
                                       error_message=f'Error processing file: {str(e)}')

    return render_template('bot_iot.html',
                           prediction_table=None,
                           total_records=0,
                           attack_records=0,
                           normal_records=0,
                           attack_percentage=0,
                           normal_percentage=0)



@app.route('/kddcup', methods=['GET', 'POST'])
def kddcup():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '': 
            return redirect(request.url)
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                if file.filename.endswith('.csv'):
                    data = pd.read_csv(filepath)
                elif file.filename.endswith('.xls'):
                    data = pd.read_excel(filepath, engine='xlrd')
                elif file.filename.endswith('.xlsx'):
                    data = pd.read_excel(filepath, engine='openpyxl')
                else:
                    return "Unsupported file format. Please upload a CSV or Excel file."

                input_table = data.to_html(classes='data')

                predictions = model_kdd.predict(data)
                data['Prediction'] = ['Normal' if pred == 1 else 'Attack' for pred in predictions]

                total_records = len(predictions)
                attack_records = np.sum(predictions == 0)
                normal_records = total_records - attack_records
                attack_percentage = (attack_records / total_records) * 100 if total_records > 0 else 0
                normal_percentage = (normal_records / total_records) * 100 if total_records > 0 else 0

                # Convert np.int32 to native Python int
                prediction_table = data.to_html(classes='data')
                print(prediction_table)
                return render_template('nslkdd.html',
                                       input_table=input_table,
                                       prediction_table=prediction_table,
                                       filename=file.filename,
                                       total_records=total_records,
                                       attack_records=int(attack_records),  # Ensure conversion
                                       normal_records=int(normal_records),  # Ensure conversion
                                       attack_percentage=attack_percentage,
                                       normal_percentage=normal_percentage)

            except Exception as e:
                return f"Error processing file: {str(e)}"

    return render_template('nslkdd.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)




@app.route('/home1')
def home_page():
    return render_template('home1.html')

@app.route('/home2')
def home1():
    return render_template('home2.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final4 = [np.array(int_features)]
    model = joblib.load('model_kdd.sav')
    predict = model.predict(final4)
    
    output = 'There is an Attack Detected in the Cloud!' if predict == 0 else 'There is No Attack Detected in the Cloud!'
    return render_template('home1.html', output=output)

@app.route('/predict2', methods=['POST'])
def predict2():
    # Collecting the form inputs and converting them to floats
    int_features = [float(x) for x in request.form.values()]
    
    # If your model expects only 15 features, extract the first 15 features
    # Adjust this part if your model expects a specific subset of features
    int_features = int_features[:15]  # Ensure only 15 features are passed
    
    # Convert the features into a numpy array to match the expected input format
    final4 = [np.array(int_features)]
    
    # Loading the trained model (adjust the file name if necessary)
    model = joblib.load('model_botiot.sav')
    
    # Making the prediction
    predict = model.predict(final4)
    
    # Output based on the prediction (assuming 0 is for attack detected)
    output = 'There is an Attack Detected in the Cloud!' if predict == 0 else 'There is No Attack Detected in the Cloud!'
    
    # Rendering the output to the home2.html template
    return render_template('home2.html', output=output)



# Route for Bot_IoT2
@app.route('/Bot_IoT2')
def bot_iot():
    return render_template('Bot_IoT2.html')

# Route for NSL-KDD
@app.route('/NSL-KDD')
def kdd_cup():
    return render_template('NSL-KDD.html')



if __name__ == '__main__':
    app.run(debug=True)
