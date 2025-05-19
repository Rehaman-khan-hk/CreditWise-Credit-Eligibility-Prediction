CreditWise: Credit Eligibility Prediction
 
CreditWise is a machine learning-based web application designed to predict credit eligibility for loan applicants. Using a Random Forest model trained on banking data, it classifies applicants into four risk categories (P1–P4, from highly eligible to not eligible). The project includes exploratory data analysis (EDA), robust data preprocessing, model training, and a user-friendly Streamlit web app for real-time eligibility checks.
Table of Contents

Project Overview
Key Features
Highlights of Work Done
Installation
Usage
Project Structure
Technologies Used
Contributing
License
Contact

Project Overview
CreditWise leverages a dataset of 51,336 loan applicants with 87 features (e.g., Credit_Score, NETMONTHLYINCOME, enq_L3m) to predict credit eligibility. The pipeline includes:

Exploratory Data Analysis (EDA): Analyzed feature distributions and correlations.
Preprocessing: Handled missing values, outliers, and encoding to prepare data for modeling.
Model Training: Compared five models (Logistic Regression, SVM, Naïve Bayes, Random Forest, DNN), with Random Forest achieving the best performance (Accuracy=98.98%, AUC=99.94%).
Prediction: Applied the model to an unseen dataset (100 rows) and saved predictions.
Web App: Built a Streamlit app for users to input details and check eligibility.

The project aligns with banking industry standards, emphasizing robust preprocessing, model interpretability, and user accessibility.
Key Features

Accurate Predictions: Random Forest model with 98.98% accuracy and 99.94% AUC.
Comprehensive EDA: Visualized Approved_Flag distribution (P2=62.72%) and identified key features like Credit_Score (importance=0.466).
Robust Preprocessing: Handled missing values (dropped 8 columns with >10,000 missing), removed outliers (reduced dataset to 31,949 rows), and one-hot encoded categoricals (90 columns).
Interactive Web App: Streamlit interface for users to input 12 features (e.g., Credit_Score, MARITALSTATUS) and receive eligibility results (P1–P4) with explanations.
Scalable Deployment: Ready for Streamlit Cloud or local hosting with ngrok.

Highlights of Work Done
The following are the critical achievements in the CreditWise project:

Exploratory Data Analysis (EDA):

Distribution Analysis: Identified class imbalance (Approved_Flag: P2=62.72%, P1=11.30%) using bar and pie charts.
Feature Insights: Found Credit_Score (mean=679.86, std=20.50) and Tot_Active_TL (mean=2.09, skewed) as key predictors.
Correlation Detection: Uncovered perfect correlations (1.0) among inquiry features (e.g., CC_enq_L6m vs. CC_enq_L12m), highlighting redundancy.
Feature Importance: Random Forest analysis ranked Credit_Score (0.466), Age_Oldest_TL (0.052), and enq_L3m (0.047) as top features.


Data Preprocessing:

Missing Values: Dropped 8 columns (e.g., CC_utilization, PL_utilization) with >10,000 missing values and imputed others with column means.
Outlier Removal: Removed 37% of rows (51,336 → 31,949) using ±4 std rule, ensuring data quality.
Encoding & Scaling: One-hot encoded categoricals (e.g., MARITALSTATUS, EDUCATION) to 90 columns and applied min-max scaling (e.g., Credit_Score mean=0.495 post-scaling).
Stratified Split: Maintained class distribution (P2=73%) in train (25,559 rows) and test (6,390 rows) sets.


Model Development:

Model Comparison: Evaluated five models:
Random Forest: Best performer (Accuracy=98.98%, AUC=99.94%, Training Time=3.01s).
DNN: Strong AUC (99.61%) with stable training (validation accuracy=95.54%).
Logistic Regression & SVM: High accuracy (~95.93%) but slower training.
Naïve Bayes: Poor performance (Accuracy=34.87%), unsuitable for correlated features.


Training Stability: DNN achieved consistent improvement over 30 epochs (final validation loss=0.1140).


Unseen Dataset Prediction:

Processed a 100-row unseen dataset (42 features → 90 columns after preprocessing).
Random Forest provided diverse predictions (P4, P2, P3), outperforming uniform predictions from other models (e.g., all P4 for Logistic Regression).
Saved predictions to unseen_predictions.csv.


Streamlit Web App:

Developed an interactive interface for users to input 12 key features (e.g., Credit_Score, NETMONTHLYINCOME, EDUCATION).
Implemented preprocessing to match training pipeline (imputation, encoding, scaling).
Displayed eligibility results (P1–P4) with user-friendly messages and key factors (e.g., “High Credit Score boosts eligibility”).
Enabled local testing with ngrok in Colab and prepared for Streamlit Cloud deployment.



Installation
Prerequisites

Python 3.11
Google Colab (for initial setup) or local environment
Git

Steps

Clone the Repository:
git clone https://github.com/your-username/CreditWise.git
cd CreditWise


Install Dependencies:
pip install -r requirements.txt

requirements.txt includes:
streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
scikit-learn==1.5.0
pyngrok==7.2.0


Prepare Model Artifacts:

Ensure the following files are in the models/ directory:
random_forest_model.pkl
scaler.pkl
X_train_columns.pkl
X_train_means.pkl
columns_to_drop.pkl


These are generated during model training (see Usage).


Set Up ngrok (Optional for Colab):

Sign up at ngrok.com and get an authtoken.
Configure in Colab:!ngrok authtoken YOUR_NGROK_AUTH_TOKEN





Usage
Running the Streamlit App Locally

Start the Streamlit server:streamlit run app.py


Open the URL (e.g., http://localhost:8501) in your browser.
Enter details in the form (e.g., Credit_Score=750, NETMONTHLYINCOME=75,000).
Click “Check Eligibility” to view the prediction (e.g., “Eligible (P2)”) and key factors.

Running in Google Colab

Upload app.py and model artifacts to /content/drive/MyDrive/cibil_score/.
Run the following in a Colab cell:!pip install streamlit pyngrok
!ngrok authtoken YOUR_NGROK_AUTH_TOKEN
!streamlit run /content/drive/MyDrive/cibil_score/app.py &>/dev/null &
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"Streamlit app running at: {public_url}")


Open the ngrok URL (e.g., https://1234-34-56-78-90.ngrok.io).
Test with sample inputs.

Example Inputs

Eligible Profile:
Credit_Score: 750
NETMONTHLYINCOME: 75,000
Age_Oldest_TL: 120
Tot_Missed_Pmnt: 0
enq_L3m: 0
MARITALSTATUS: Married
Expected: P1 or P2


Not Eligible Profile:
Credit_Score: 500
NETMONTHLYINCOME: 20,000
Tot_Missed_Pmnt: 3
enq_L3m: 5
Expected: P3 or P4



Project Structure
CreditWise/
├── app.py                  # Streamlit web app
├── models/                 # Model artifacts
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│   ├── X_train_columns.pkl
│   ├── X_train_means.pkl
│   ├── columns_to_drop.pkl
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── data/                   # Optional: Sample data (not included)

Technologies Used

Python Libraries:
pandas, numpy: Data manipulation
scikit-learn: Model training and preprocessing
streamlit: Web app interface
joblib: Model serialization
pyngrok: Local tunneling for Colab


Machine Learning:
Random Forest, Logistic Regression, SVM, Naïve Bayes, DNN (Keras)


Environment:
Google Colab (Python 3.11)
GitHub for version control



Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/new-feature).
Commit changes (git commit -m 'Add new feature').
Push to the branch (git push origin feature/new-feature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact

Author: Rehaman Khan H.K
Email: rehamankhan.1si23lvs10@gmail.com



Disclaimer: CreditWise is an indicative tool for credit eligibility. For official decisions, contact your bank.
