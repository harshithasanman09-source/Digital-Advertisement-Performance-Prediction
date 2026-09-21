# Digital-Advertisement-Performance-Prediction
# Digital Advertisement Performance Prediction  A machine learning project that predicts the performance of digital advertising campaigns as **Low, Medium, or High** based on campaign characteristics and engagement metrics.  The project is developed using **Python and Scikit-learn** and uses a Random Forest Classifier with preprocessing 
# Digital Advertisement Performance Prediction

A machine learning project that predicts the performance of digital advertising campaigns as **Low, Medium, or High** based on campaign characteristics and engagement metrics.

The project is developed using **Python and Scikit-learn** and uses a Random Forest Classifier with preprocessing for categorical features.

## Project Overview

Digital advertising campaigns generate different levels of performance depending on factors such as advertising platform, advertisement type, budget, impressions, click-through rate, engagement rate, target audience, and campaign duration.

This project uses these campaign attributes to build a classification model capable of predicting the expected performance category of an advertisement campaign.

## Objectives

* Predict digital advertisement performance.
* Classify campaigns into Low, Medium, and High performance categories.
* Apply machine learning preprocessing techniques to categorical and numerical data.
* Train and evaluate a Random Forest classification model.
* Generate performance predictions for new advertising campaigns.
* Evaluate the model using accuracy, classification report, and confusion matrix.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib

## Machine Learning Algorithm

### Random Forest Classifier

The project uses a Random Forest Classifier with:

* 250 decision trees
* `random_state=42`
* Balanced class weights

Random Forest combines multiple decision trees to improve classification performance and handle different types of input features.

## Dataset

The project uses a synthetic digital advertisement dataset containing **1,600 records**.

### Input Features

| Feature            | Description                                       |
| ------------------ | ------------------------------------------------- |
| Platform           | Digital advertising platform                      |
| Ad Type            | Type of advertisement                             |
| Budget             | Campaign advertising budget                       |
| Impressions        | Number of times the advertisement was displayed   |
| Click Through Rate | Percentage of users who clicked the advertisement |
| Engagement Rate    | User engagement percentage                        |
| Target Audience    | Age group targeted by the campaign                |
| Campaign Duration  | Duration of the campaign in days                  |

### Target Variable

`performance_class`

Possible values:

* Low
* Medium
* High

## Data Preprocessing

The project uses `ColumnTransformer` and `OneHotEncoder` to process categorical features.

Categorical features:

* Platform
* Advertisement Type
* Target Audience

Numerical features are passed through without transformation.

The preprocessing and Random Forest model are combined into a Scikit-learn Pipeline.

## Project Structure

```text
Digital_Advertisement_Performance_Prediction_Sklearn/
│
├── data/
│   └── digital_advertisement_dataset.csv
│
├── confusion_matrix.png
├── digital_advertisement_performance_model.pkl
├── predict.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Model Training

The dataset is divided into training and testing sets using an 80:20 split.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The model is then trained using the training data and evaluated using the test data.

## Model Evaluation

The project evaluates the trained model using:

* Accuracy Score
* Classification Report
* Confusion Matrix

The confusion matrix is automatically generated and saved as:

```text
confusion_matrix.png
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Digital_Advertisement_Performance_Prediction_Sklearn
```

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train_model.py
```

This will:

1. Load the dataset.
2. Separate input features and target variable.
3. Encode categorical features.
4. Split the dataset into training and testing sets.
5. Train the Random Forest model.
6. Generate predictions.
7. Display accuracy and classification results.
8. Save the trained model.
9. Generate the confusion matrix.

The trained model will be saved as:

```text
digital_advertisement_performance_model.pkl
```

## Make Predictions

Run:

```bash
python predict.py
```

The prediction script uses a sample advertising campaign containing information such as:

```text
Platform: Google
Ad Type: Video
Budget: 85000
Impressions: 240000
Click Through Rate: 8.4
Engagement Rate: 16.5
Target Audience: 25-34
Campaign Duration: 14 days
```

The model returns the predicted performance category along with the probability for each class.

## Example Output

```text
Predicted Advertisement Performance: High

Class probabilities:
High: XX.XX%
Low: XX.XX%
Medium: XX.XX%
```

The exact probabilities depend on the trained model.

## Key Features

* End-to-end machine learning workflow
* Categorical feature encoding
* Random Forest classification
* Scikit-learn Pipeline
* Train/test data splitting
* Class-balanced model training
* Model persistence using Joblib
* Classification report generation
* Confusion matrix visualization
* Prediction script for new campaign data

## Applications

This type of system can be used as a foundation for:

* Digital marketing analytics
* Advertisement campaign analysis
* Marketing performance prediction
* Campaign planning
* Customer targeting analysis
* Data-driven advertising decisions

## Limitations

The dataset included in this project is synthetic and intended for academic and educational purposes. The model should not be considered a production advertising-performance forecasting system without validation using real-world campaign data.

## Future Improvements

Possible improvements include:

* Testing additional machine learning algorithms
* Hyperparameter tuning
* Feature importance analysis
* Cross-validation
* Adding real-world advertising datasets
* Building an interactive Streamlit interface
* Adding data visualization dashboards
* Deploying the prediction model as a web application
* Adding automated model retraining

## Author

Developed as a machine learning project for academic and educational purposes.
