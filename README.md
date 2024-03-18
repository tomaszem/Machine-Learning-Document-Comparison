# Machine Learning Document Comparison
## Overview
This project, "Machine Learning Document Comparison", is focused on developing algorithms and methods for comparing documents using machine learning techniques. The aim is to implement and refine various algorithms and clustering methods to enable efficient and accurate comparison of textual data.

## Installation

### Docker Compose
To build and run Docker environment using Docker Compose, use the following command:

```bash
$ docker-compose up -d --build
```
This command performs the following actions:

`--build` Forces the build of the Docker images even if an image exists.

`-d` Runs the containers in the background (detached mode), allowing you to continue using the terminal for other tasks.
.

### Only Back-End app
To build the Image only with Back-End app using the following command:

```bash
$ docker build -t clustering-app:latest .
```

Run the Docker Back-End app container using the command shown below:

```bash
$ docker run -d -p 5000:5000 clustering-app
```