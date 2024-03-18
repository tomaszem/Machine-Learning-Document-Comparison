from flask import Flask
from flask import request
from flask_cors import CORS
import os
from app.prepare_json_data import prepare_json_data
from flask import jsonify
from app.clustering_operations import perform_clustering
from datetime import datetime
import json
from apscheduler.schedulers.background import BackgroundScheduler
import glob

app = Flask(__name__)
CORS(app)  # Allow CORS

@app.route('/')
def home():
    return 'Home'


@app.route('/next-run')
def next_run():
    jobs = scheduler.get_jobs()
    if jobs:
        next_run_time = jobs[0].next_run_time
        now = datetime.now(next_run_time.tzinfo)
        remaining_minutes = (next_run_time - now).total_seconds() / 60
        return jsonify({"minutes_until_next_run": round(remaining_minutes, 2)})
    else:
        return jsonify({"Error": "No scheduled jobs found."})


def scheduled_cluster():
    # Perform clustering operations and retrieve the results
    filenames, reduced_vectors, clusters = perform_clustering()

    # Prepare data for JSON
    json_data = prepare_json_data(filenames, reduced_vectors, clusters)

    # SAVE
    filename = f"cluster_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"Cluster data updated and saved to {filename}")


@app.route('/push-data')
def cluster():
    # Perform clustering operations and retrieve the results
    filenames, reduced_vectors, clusters = perform_clustering()

    # Prepare data for JSON
    json_data = prepare_json_data(filenames, reduced_vectors, clusters)

    # Use jsonify to convert data to JSON and return it with the correct content type
    return jsonify(json_data)


@app.route('/get-data')
def get_data():
    list_of_files = glob.glob('./*.json')
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            return jsonify(json.load(f))
    else:
        return jsonify({"error": "No JSON files found."})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.pdf'):
        # Define the base directory dynamically
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the target directory for PDF files
        documents_dir = os.path.join(base_dir, 'documents', 'pdf')

        if not os.path.exists(documents_dir):
            os.makedirs(documents_dir)

        # Construct the full path
        file_path = os.path.join(documents_dir, file.filename)

        # Save the file
        file.save(file_path)

        return jsonify({'message': 'File uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_cluster, 'cron', hour=2, minute=00)
    scheduler.start()

    port = int(os.environ.get("PORT", 5000))
    # use_reloader=False important for APScheduler
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
