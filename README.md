# AutomaticTraining-DataCommit

This repository is part of an article series about Automatic Training with a CI/CD MLOps approach. These files are in charge of retraining an existing model (at Google Cloud Storage) whenever a push is detected into the AutomaticTraining-Dataset repository, that way we guarantee the Continuous Integration.

FYI: This project is orchestrated with Jenkins and Kubernetes.

You can find the dataset repository [here](https://github.com/sergiovirahonda/AutomaticTraining-Dataset)
The AutomaticTraining-CodeCommit repository [here](https://github.com/sergiovirahonda/AutomaticTraining-CodeCommit) - which is the code used to train a model from scratch whenever a push in the dataset repo has been detected.  Also, this repo is in charge of auto-tweaking the model if it doesn't reach a desired metric. Once the training ends, it pushes the model to GCS.
The AutomaticTraining-UnitTesting repository [here](https://github.com/sergiovirahonda/AutomaticTraining-UnitTesting) - which is in charge of testing the resulting model in a clone of the production environment. It's a Flask based web app that is intended to run on Kubernetes to load the model from GCS, test it and inform the results.
The AutomaticTraining-PredictionAPI repository [here](https://github.com/sergiovirahonda/AutomaticTraining-PredictionAPI) is the prediction service API and returns the predictions as JSON responses.
The AutomaticTraining-Deployment repository [here](https://github.com/sergiovirahonda/AutomaticTraining-Deployment) is a Python script used to semi-automate the deployment to production. It copies the model from the model testing registry at GCS to the production one and systematically shut down the Kubernetes pods of PredictionAPI service to load the new production model.
The AutomaticTraining-Interface repository [here](https://github.com/sergiovirahonda/AutomaticTraining-Interface) is the web-interface of the project.


Feel free to fork them if it's of your interest.
