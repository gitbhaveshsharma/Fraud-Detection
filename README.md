# Fraud Detection Project :money_with_wings: :detective:

## Dataset Information :clipboard:

- Dataset Name: Fraud.csv
- Total Entries: 6,362,620
- Columns: 11
  - step: int64
  - type: object
  - amount: float64
  - nameOrig: object
  - oldbalanceOrg: float64
  - newbalanceOrig: float64
  - nameDest: object
  - oldbalanceDest: float64
  - newbalanceDest: float64
  - isFraud: int64
  - isFlaggedFraud: int64

## Project Overview :mag_right:

This project focuses on fraud detection in financial transactions using machine learning algorithms. The goal is to build a model that can accurately identify fraudulent transactions based on the available features. :credit_card: :moneybag:

## Libraries Used :books:

- pandas
- numpy
- sklearn.preprocessing (LabelEncoder, OneHotEncoder, StandardScaler)
- seaborn
- matplotlib.pyplot
- sklearn.model_selection (train_test_split, cross_val_score, GridSearchCV)
- statsmodels.api
- statsmodels.stats.outliers_influence (variance_inflation_factor)
- sklearn.ensemble (RandomForestClassifier)
- sklearn.linear_model (LogisticRegression)
- xgboost
- sklearn.metrics (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer)

## Steps Performed :chart_with_upwards_trend:

1. Data Loading: The dataset "Fraud.csv" was loaded using pandas. :open_file_folder:
2. Data Preprocessing: 
   - Label Encoding: The categorical features (type, nameOrig, nameDest) were label-encoded to convert them into numeric form. :1234:
   - One-Hot Encoding: The label-encoded features were further one-hot encoded to create binary columns for each unique value. :arrows_counterclockwise:
   - Scaling: The numeric features were scaled using StandardScaler to normalize their values. :balance_scale:
3. Feature Engineering: Additional features such as transaction frequency, average transaction amount, hour of day, day of week, and month of year were derived from the existing features. :gear:
4. Train-Test Split: The dataset was split into training and testing sets to evaluate the model's performance on unseen data. :train:
5. Model Selection: Three machine learning models were selected for fraud detection:
   - XGBoost
   - Random Forest
   - Logistic Regression
6. Model Training and Evaluation: Each model was trained on the training data and evaluated using various performance metrics, including accuracy, precision, recall, F1 score, and ROC AUC score. :bar_chart:
7. Model Optimization: Hyperparameter tuning was performed to optimize the model's performance using techniques like cross-validation and grid search. :dart:
8. Validation and Further Testing: The optimized models were further validated using techniques like cross-validation or holdout validation to ensure consistent and robust performance. :white_check_mark:
9. Model Comparison: The performance of each model was compared, and the model with the best performance metrics was selected. :chart_with_downwards_trend:
10. Model Deployment: The selected model can be deployed for real-time fraud detection in financial transactions. :rocket:

## Results :chart_with_upwards_trend:

The performance metrics for each model are as follows:

### XGBoost :rocket:
- Accuracy: 99.98% :white_check_mark:
- Precision: 97.64% :dart:
- Recall: 89.51% :eyes:
- F1 Score: 93.39% :balance_scale:
- ROC AUC Score: 94.75% :chart_with_upwards_trend:

### Random Forest :deciduous_tree:
- Accuracy: 99.98% :white_check_mark:
- Precision: 99.33% :dart:
- Recall: 81.98% :eyes:
- F1 Score: 89.82% :balance_scale:
- ROC AUC Score: 90.99% :chart_with_upwards_trend:

### Logistic Regression :chart_with_downwards_trend:
- Accuracy: 70.58% :warning:
- Precision: 0.33% :warning:
- Recall: 77.47% :eyes:
- F1 Score: 93.39% :balance_scale:
- ROC AUC Score: 74.02% :chart_with_downwards_trend:

Based on the performance metrics, both XGBoost and Random Forest outperform Logistic Regression in terms of accuracy, precision, recall, F1 score, and ROC AUC score. Therefore, either XGBoost or Random Forest can be chosen as the preferred model for fraud detection. :raised_hands:

Note: The performance of the models may vary depending on the specific dataset, model configuration, and data preprocessing techniques used. Regular monitoring and evaluation of the model's performance are necessary to ensure its effectiveness in detecting fraud and adapt to changing patterns in fraudulent activities. :hourglass_flowing_sand:

License :page_with_curl:
This project is licensed under the MIT License. You are free to use, modify, or distribute this project for educational or commercial purposes.
