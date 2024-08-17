import subprocess
import webbrowser
import time


def run_flask_app_and_open_browser():
    # Start the Flask app in a separate process
    flask_process = subprocess.Popen(['python', 'app.py'])

    # Wait for the Flask app to start
    time.sleep(2)

    # Open the web browser to the index page
    webbrowser.open('http://127.0.0.1:5000/')

    print("Flask app is running. You can now use the following endpoints:")
    print("- POST /submit-data: Submit new data")
    print("- GET /get-data: Retrieve all data")
    print("- POST /predict: Make a flood prediction")
    print("- GET /retrain: Retrain the model with all data")

    # Wait for user input before closing
    input("Press Enter to close the application...")

    # Terminate the Flask process
    flask_process.terminate()


if __name__ == '__main__':
    run_flask_app_and_open_browser()