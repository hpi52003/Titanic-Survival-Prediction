# Titanic-Survival-Prediction
## Overview
This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The dataset used for this analysis is the famous Titanic dataset, which contains details about passengers such as age, gender, ticket class, and whether they survived.

## Objectives
- Perform data preprocessing and exploratory data analysis (EDA)
- Apply feature engineering to improve model performance
- Train machine learning models to predict passenger survival
- Evaluate model performance and optimize hyperparameters

## Requirements
Ensure you have the following installed:
- Python (>=3.8)
- Jupyter Notebook or any Python IDE
- Required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

## Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```
2. Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data) and place it in the project directory.
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Titanic_Survival_Prediction.ipynb
   ```
4. Run the notebook to preprocess the data, train models, and generate predictions.
5. Modify and test different models to improve accuracy.

## Project Structure
```
├── data/
│   ├── train.csv
│   ├── test.csv
├── notebooks/
│   ├── Titanic_Survival_Prediction.ipynb
├── models/
│   ├── saved_model.pkl
├── README.md
├── requirements.txt
├── main.py
```

## Machine Learning Models Used
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Classifier

## Evaluation Metrics
- Accuracy Score
- Precision, Recall, and F1-Score
- Confusion Matrix

## Contributing
Feel free to submit pull requests or report issues. Contributions are welcome!

## License
This project is licensed under the MIT License.

