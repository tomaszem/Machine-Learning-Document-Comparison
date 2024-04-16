# Scientific Papers Clustering
## Overview
The project aims to cluster scientific papers using fast and effective and fast algorithms such as TF-IDF, PCA/UMAP, DBSCAN etc. The Back-End application also analyzes other relevant information from scientific documents. Built on Python Flask, this back-end application leverages Chroma DB as a vector database for efficient data handling and retrieval. Automatic clustering is facilitated through Advanced Python Scheduler, allowing scheduling of clustering processes by time and date.


## Installation

### Docker Compose
To build and run Docker environment using Docker Compose, use the following command:

```bash
$ docker-compose up -d --build
```
This command performs the following actions:

`--build` Forces the build of the Docker images even if an image exists.

`-d` Runs the containers in the background (detached mode), allowing you to continue using the terminal for other tasks.

This option below forces Docker Compose to recreate the containers whether or not there are changes in the configuration.
```bash
$ docker-compose up -d --build --force-recreate
```
### Only Back-End app
To build the Image only with Back-End app using the following command:

```bash
$ docker build -t clustering-app:latest .
```

Run the Docker Back-End app container using the command shown below:

```bash
$ docker run -d -p 5000:5000 clustering-app
```