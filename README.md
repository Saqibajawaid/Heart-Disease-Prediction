# Heart Disease Prediction Using Machine Learning

## Project Overview

This project uses machine learning classification algorithms to predict the presence or absence of heart disease based on selected patient-related attributes. It includes data analysis, visualization, feature selection, model training, model evaluation, and an interactive web application built with Streamlit.

**Disclaimer:** This project is for educational purposes only. Its predictions are not medical advice and must not be used for diagnosis or treatment decisions.

## Dataset

The project uses the **Heart Disease Cleveland UCI** dataset (`heart_cleveland_upload.csv`).

- **Records:** 297
- **Columns:** 14
- **Input features:** 13
- **Target column:** `condition`
- `condition = 0`: No heart disease in the dataset
- `condition = 1`: Heart disease present in the dataset

### Input Features

| Feature | Description |
|---|---|
| `age` | Age |
| `sex` | Sex (numerically encoded) |
| `cp` | Chest pain type (numerically encoded) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting electrocardiographic results (numerically encoded) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (numerically encoded) |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia-related category (numerically encoded) |

The numerical category codes should be interpreted according to the dataset documentation.

## Project Workflow

1. **Data Loading** – Load the dataset using Pandas.
2. **Understanding Data** – Examine the dataset structure, data types, statistical summary, and target distribution.
3. **Data Preprocessing** – Prepare the dataset for analysis and model development.
4. **Exploratory Data Analysis (EDA)** – Explore feature distributions and relationships using visualizations.
5. **Feature and Target Separation** – Separate input features (`X`) from the target (`y`).
6. **Train-Test Split** – Divide the data into 70% training and 30% testing sets using stratification.
7. **Feature Selection** – Use SelectKBest with the ANOVA F-test to select the top 10 features.
8. **Model Selection and Training** – Train five classification algorithms.
9. **Model Evaluation** – Compare models using Accuracy, Precision, Recall, F1 Score, a Confusion Matrix, and a Classification Report.
10. **Web Application** – Build an interactive prediction interface with Streamlit.

## Machine Learning Algorithms

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

The models are compared using evaluation metrics. The best-performing model is selected based on the F1 Score.

## Technologies and Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Project Structure

A suggested project structure is:

```text
heartDiseaseprediction/
│
├── HDP.ipynb
├── app.py
├── heart_cleveland_upload.csv
├── heart_disease_model.pkl
├── feature_selector.pkl
└── README.md
```

The `.pkl` files are generated after model training. Your actual project folder may contain additional files, such as saved graphs or environment folders.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-link>
cd heartDiseaseprediction
```

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create and Activate a Virtual Environment (Optional)

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib jupyter
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `HDP.ipynb` and run the notebook cells in order.

### 5. Run the Streamlit Application

Make sure `app.py`, `heart_disease_model.pkl`, and `feature_selector.pkl` are in the expected project locations. Then run:

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal. Open that address in your browser to use the application.

## Model Evaluation

The models are evaluated on the held-out testing dataset using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1 Score:** Harmonic mean of Precision and Recall.
- **Confusion Matrix:** Displays correct and incorrect predictions by class.
- **Classification Report:** Provides class-wise evaluation metrics.

Refer to the model comparison results in `HDP.ipynb` for the measured scores.

## Limitations

- The project uses a relatively small dataset.
- The dataset contains numerically encoded categorical features.
- Performance on this dataset may not generalize to other populations or real-world clinical settings.
- Predictions are educational outputs and are not a substitute for professional medical assessment.

## Future Improvements

- Test the models with additional and more diverse datasets.
- Explore hyperparameter tuning.
- Improve the Streamlit interface and input validation.
- Add clearer explanations of model predictions.
- Evaluate generalization using additional validation techniques.

## Author

**Saqiba Jawaid**  
BS Information Technology Student

---

*Developed as an academic machine learning project.*
