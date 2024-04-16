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
    documentLocation = constants.PDF_PATH
    if not documentLocation:
        return jsonify({'error': 'No document location provided'}), 400

    try:
        ChromaDBClient.update_collection_chromadb(documentLocation)
        return jsonify({'message': 'Done'}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to update collection', 'exception': str(e)}), 500


# @app.route('/cluster-data')
# def cluster():
#     # Perform clustering operations and retrieve the results
#     filenames, reduced_vectors_3d, initial_clusters, reduced_vectors_2d, final_clusters = perform_clustering()
#
#     # Prepare data for JSON
#     json_data = prepare_json_data_v2(filenames, reduced_vectors_3d, initial_clusters, reduced_vectors_2d,
#                                      final_clusters)
#
#     return jsonify(json_data)


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
        return jsonify({'message': 'Done'}), 200
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

        return jsonify({'message': 'File uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400


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
        print(f"Received EPS value: {eps_value}")

        # Define the path to the config.yaml file
        config_path = os.path.join('app', 'config', 'config.yaml')

        # Load the current contents of the config.yaml file
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file) or {}
        else:
            config = {}

        # Update the EPS value
        if 'dbscan' not in config:
            config['dbscan'] = {}
        config['dbscan']['eps'] = eps_value

        # Write the updated dictionary back to the config file
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file, default_flow_style=False)

        return jsonify({"message": "Success", "eps": eps_value}), 200
    else:
        return jsonify({"error": "Internal error"}), 400


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_cluster, 'cron', hour=2, minute=00)
    scheduler.start()

    port = int(os.environ.get("PORT", 5000))
    # use_reloader=False important for APScheduler
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
