from flask import Flask
from flask import request
from flask_cors import CORS
import os
from flask import jsonify
from datetime import datetime
import json
from apscheduler.schedulers.background import BackgroundScheduler
import glob
import numpy as np
import yaml
from ChromaDB import ChromaDBClient
from app.clustering_operations import perform_clustering
from app.prepare_json_data import prepare_json_data
from app.optimal_eps_range import find_optimal_eps_range
from app.pdf_info_extraction import get_pdf_details
from app.prepare_json_pdf_data import pdf_info_to_json
from app.prepare_json_data import prepare_json_data_v2
import app.config.constants as constants
import platform

app = Flask(__name__)
CORS(app)  # Allow CORS
scheduler = BackgroundScheduler()
job = None


@app.route('/')
def home():
    python_info = {
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
    }

    health_info = {
        "status": "up",
        "current_time": datetime.now().isoformat(),
        "os_info": platform.version(),
        "python_info": python_info
    }

    return jsonify(health_info)


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
    documentLocation = constants.PDF_PATH
    if not documentLocation:
        return jsonify({'error': 'No document location provided'}), 400

    try:
        ChromaDBClient.update_collection_chromadb(documentLocation)
        return jsonify({'message': 'Done'}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to update collection', 'exception': str(e)}), 500


@app.route('/pdf-data')
def get_pdf_detail():
    pdf_details = get_pdf_details()
    pdf_details_json = pdf_info_to_json(pdf_details)
    return jsonify(pdf_details_json)


@app.route('/get-data')
def get_data():
    collection = ChromaDBClient.get_collection()
    if collection is None:
        return jsonify({"error": "No collection found."}), 400
    return jsonify(collection)


@app.route('/cluster-data', methods=['POST', 'GET'])
def cluster():
    documentLocation = constants.PDF_PATH
    if not documentLocation:
        return jsonify({'error': 'No document location provided'}), 400

    try:
        ChromaDBClient.update_collection_chromadb(documentLocation)
        return jsonify({'message': 'Success'}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to update collection', 'exception': str(e)}), 500


@app.route('/upload-pdf', methods=['POST'])
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

        return jsonify({"message": "Success"}), 200
    else:
        return jsonify({"error": "Internal error"}), 400


@app.route('/get-eps-range')
def get_eps_range():
    collection = ChromaDBClient.get_collection()
    if collection is None:
        return jsonify({"error": "No collection found."}), 400

    jsonify(collection)
    x_values = []
    y_values = []

    for metadata in collection['metadatas']:
        reduce_vectors_2d = json.loads(metadata['reduce_vectors_2d'])
        x_values.append(reduce_vectors_2d[0][0])
        y_values.append(reduce_vectors_2d[0][1])

    data_array = np.array([[x, y] for x, y in zip(x_values, y_values)])

    # Find the optimal EPS range
    start_eps, end_eps, suggested_eps_values = find_optimal_eps_range(data_array)

    result = {
        "start_eps": start_eps,
        "end_eps": end_eps,
        "suggested_eps_values": suggested_eps_values
    }

    return jsonify(result)


@app.route('/submit-eps', methods=['POST'])
def submit_eps():
    data = request.get_json()
    eps_value = data.get('eps')

    if eps_value is not None:
        config_path = os.path.join('app', 'config', 'config.yaml')

        # Load the current contents of the config.yaml file
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file) or {}
        else:
            config = {}

        if 'dbscan' not in config:
            config['dbscan'] = {}
        config['dbscan']['eps'] = eps_value

        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)

        return jsonify({"message": "Success"}), 200
    else:
        return jsonify({"error": "Internal error"}), 400


@app.route('/submit-clusters-num', methods=['POST'])
def receive_clusters_num():
    data = request.get_json()
    clusters_num = data.get('numberOfClusters')

    if clusters_num is not None:
        config_path = os.path.join('app', 'config', 'config.yaml')

        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file) or {}
        else:
            config = {}

        if 'agglomerative_clusters' not in config:
            config['agglomerative_clusters'] = {}
        config['agglomerative_clusters']['n'] = None if clusters_num == 0 else clusters_num

        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)

        return jsonify({"message": "Success"}), 200
    else:
        return jsonify({"error": "Internal error"}), 400


@app.route('/submit-cron-time', methods=['POST'])
def submit_cron():
    global job
    hours = request.json['hours']
    minutes = request.json['minutes']
    if hours is not None and minutes is not None:
        if job:
            scheduler.remove_job(job.id)

        job = scheduler.add_job(scheduled_cluster, 'cron', hour=hours, minute=minutes)
        return jsonify({"message": "Success"}), 200
    else:
        return jsonify({"error": "Internal error"}), 400


if __name__ == "__main__":
    scheduler.start()
    job = scheduler.add_job(scheduled_cluster, 'cron', hour=2, minute=00)

    port = int(os.environ.get("PORT", 5000))
    # use_reloader=False important for APScheduler
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
