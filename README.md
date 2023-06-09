# Backorder Prediction - Django Web Application

This is a Django web application that uses machine learning to predict whether a product will go on backorder or not. It uses a pre-trained Random Forest Classifier, Decision Tree and LGBM models to make predictions based on various features such as product availability, lead time, and more.

## ****Cloning the Project****

To clone this project, run the following command in your terminal:

```bash
git clone https://github.com/Pradyothsp/backorder-prediction.git
cd backorder-prediction
```

## Creating Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## ****Getting Started****

Navigate to the project directory and follow the steps below to set up and run the project locally:

```bash
cd app
```

1. Install the required dependencies by running the following command:
    
    ```bash
    pip install -r requirements/local.txt
    ```
    
2. Run the Django migrations to set up the database:
    
    ```bash
    python3 manage.py makemigrations
    python3 manage.py migrate
    ```
    
3. Start the Django development server:
    
    ```bash
    python3 manage.py runserver
    ```
    
4. Open your web browser and navigate to **`http://127.0.0.1:8000/`** to view the application.
    
    The application has a simple web interface where you can input the product features and get a prediction on whether the product will go on backorder or not.
    
    You can also make predictions using the API endpoint by sending a POST request to **`http://127.0.0.1:8000/predict/`** with a JSON payload containing the product features.

## ****Development****
This application was developed using Django 4.2. The backorder prediction model was trained using scikit-learn and is stored in the **`backorder/model/backorder_best_model.pkl`** file.

To train a new prediction model, you can run the `model.ipynb` Jupyter Notebook in the root directory.

## ****Credits****
This project was created by [Pradyoth S P](https://github.com/Pradyothsp/backorder-prediction).

The backorder dataset used to train the prediction model is from the [Kaggle](https://www.kaggle.com/datasets/gauravsbisht/backorder-prediction).