from flask import Flask, request, render_template
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('safetrix-117b2-firebase-adminsdk-gakry-63a5157e5f.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit-data', methods=['POST'])
def submit_data():
    data = request.form.get('data')
    print("Received data:", data)  # Debugging: Print received data

    if data:
        # Add the received data to Firestore
        db.collection('collected_data').add({'data': data})
        return "Data submitted successfully"
    else:
        return "No data received", 400

if __name__ == '__main__':
    app.run(debug=True)
