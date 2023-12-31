## Capstone project 1 for Machine Learning Zoomcamp

Hello everyone, this is my Capstone Project 1 for the for the 2023 cohort of the course [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp). 

This project will aim to train a machine learning model to predict prices of houses in a synthetic dataset.

The dataset can be found [here](https://www.kaggle.com/datasets/muhammadbinimran/housing-price-prediction-data).

To download the dataset, please run the following command:
```bash
make download-data
```
You need to have `kaggle` installed on your machine and also a kaggle account. You can get started [here](https://www.kaggle.com/docs/api).

This dataset provides a realistic yet simplified platform for developing and testing a machine learning model aimed at predicting house prices.

Why This Dataset?

1. Realistic Scenario with Managed Complexity:
The synthetic nature of this dataset mirrors real-world scenarios without the complications of missing or inconsistent data. This feature is particularly advantageous for focusing on model development and methodology, making the dataset ideal for educational and training purposes.

2. Rich and Diverse Features:
Containing key attributes like 'SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt', and 'Price', the dataset encompasses a range of factors influencing real estate pricing. This diversity allows us to delve into various data analysis, preprocessing, and feature engineering techniques.

3. Comprehensive Analytical Opportunities:
The blend of numerical and categorical data in the dataset provides a broad scope for applying different preprocessing strategies.

4. Benchmarking and Model Evaluation:
Housing price prediction is a well-established problem in machine learning, offering a solid basis for benchmarking a variety of algorithms. This project serves as an excellent platform for comparing the effectiveness of different models, tuning their hyperparameters, and exploring the intricacies of machine learning models in a practical context.

The final app can be tried here: https://streamlist-car-prices-jvgkqtvdxq-ey.a.run.app

Here is a screenshot:
![Alt text](readme_images/preview.png)

### Exploratory data analysis

The EDA can be found inside the Jupyter Notebook called `eda.ipynb` in the root directory of the project.

### Dataset building and model training.

The building of the train, valid and test datasets happens inside the `model_training.ipynb` notebook. There I also run a grid search analysis and also hyperparameter tuning in order to pick the best performing model. After training and getting the best performing model I run an analysis in order to see how well the model is doing.

After making sure the model performs in a satisfactory manner, I save the model using the `pickle` module from Python.

The notebook can also be converted to a Python file to be run like a module using the command:
```bash
jupyter nbconvert --to script model_training.ipynb
```

To run the model training, please run the following command:
```bash
python model_training.py
```
### Running the app

If you want to run the app, please first make sure to have installed `pipenv`. If you haven't already, you can do so by running `pip install pipenv`.

After having installed `pipenv`, please run `pipenv install` in order to install the dependencies needed to run the app. 

After you have installed all the dependencies, please run `pipenv run streamlit run docker_app.py` to run the app. The streamlit app should now be running in your browser.

### Docker and cloud deployment.

Inside the main directory there is a Dockerfile which can be built in order to create an image of the sreamlit app.

Simply run `docker build -t my-streamlit-app .` to build the image. Then, you can run the container by giving the command `docker run -p 8080:8080 my-streamlit-app
`.

For this project, I have chosen Google [Cloud Run](https://cloud.google.com/run). To use this service, you need to have an account on Google Cloud Platform. To get started, please visit this [website](https://cloud.google.com/). You get 300$ of free credits when getting started, which is pretty generous.

Since this project is not necessarily focused on Operations, I deployed the app mostly manually, with the help of the UI.

The first step is to download the Google Cloud CLI. You can get started [here](https://cloud.google.com/sdk/gcloud). After having installed the CLI and being able to use the command `gcloud` you can tag the image you created above to reference your Google Cloud Project. The commands would be: 
```bash
docker tag my-streamlit-app gcr.io/<google-cloud-project>/my-stream-lit-app
docker push gcr.io/<google-cloud-project>/my-stream-lit-app
```
After this, please go to Google Cloud Run and click on `Create Service`. After that you can fill in the necessary information.
![Alt text](readme_images/gcr.png)

When finishing the deployment, your app will be live and you will be provided a website link from Google (similar to the one I posted at the top of the README).