from flask import Flask
import os
from app.prepare_json_data import prepare_json_data
from flask import jsonify
from app.clustering_operations import perform_clustering
from datetime import datetime
import json
from apscheduler.schedulers.background import BackgroundScheduler
import glob

app = Flask(__name__)


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


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_cluster, 'cron', hour=2, minute=00)
    scheduler.start()

    port = int(os.environ.get("PORT", 5000))
    # use_reloader=False important for APScheduler
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
