from flask import Flask
import os
from app.prepare_json_data import prepare_json_data
from flask import jsonify
from app.clustering_operations import perform_clustering

app = Flask(__name__)


@app.route('/')
def home():
    return 'Home'


@app.route('/data')
def cluster():
    # Perform clustering operations and retrieve the results
    filenames, reduced_vectors, clusters = perform_clustering()

    # Prepare data for JSON
    json_data = prepare_json_data(filenames, reduced_vectors, clusters)

    # Use jsonify to convert data to JSON and return it with the correct content type
    return jsonify(json_data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
