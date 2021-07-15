# uwave-gesture-detection

# How to run
- Download uWaveGestureLibrary.zip to ./data
- Run dataloading/prepare_explore.ipynb to create ./data/uwave/uwave.pkl and optionally plot some samples
- Run traning/training.ipynb to train a Logistic Regression model, save it, and prepare it for deployment.

# TODO
- Docker and deploy to EBS
- Github Action to EBS

# DevOps pipeline
- On main/master pull: check the code (pep8/lint), build container, train the model, verify results, re-deploy
- Possibly run linter during development and/or pep8 before deployment
- Possibly package as pip-installable, then verify install before re-deployment

# Inference API
- Container with Gunicorn and small framework (Flask, Starlette, FastAPI) deployed on appropriate machine(s)
- To deploy, either build the model into the container (small) or have the container download the weights and configuration on startup (large)

# Larger scale
- Possibly use managed models (e.g. Sagemaker Logistic Regression) or host models on dedicated inference endpoints inside a closed network
- Privacy: Encrypt data in storage/transit, split off customer data and model access into separate networks/entrypoints
- If ETL part of the pipeline, possibly use managed services/streams like AWS Glue and Kinesis
- Think about (continuous/schedules?) re-training

# Transfer ML models to microservices
- Build into container and update container (small), or have containers check for updated weights&configs (larger)