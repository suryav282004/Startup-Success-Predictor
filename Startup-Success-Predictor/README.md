# Startup Success Probability Prediction and Business Performance Analytics Using Ensemble Machine Learning

## Project Name
**Startup Success Predictor – India Edition**

## Description
Startup Success Predictor – India Edition is an interactive machine learning application that estimates whether an Indian startup has a higher or lower probability of success using business and financial indicators. Users enter funding, industry, annual revenue, team size, revenue growth, customer growth, customer retention, monthly operating burn, cash runway, startup age, and estimated market size. All financial inputs use Indian Rupees: funding, annual revenue, and market size are measured in ₹ Crore, while monthly operating burn is measured in ₹ Lakh.

The application compares Random Forest, Extra Trees, and Logistic Regression classifiers, then combines them using a Soft Voting Ensemble. It displays model evaluation metrics, predicted success probability, business indicator observations, and Random Forest feature importance. The project uses Python, Scikit-learn, Pandas, NumPy, and Streamlit, and is designed for Streamlit Community Cloud deployment.

**Important:** The app uses synthetic demonstration data generated within the code. Its metrics and probabilities are not validated real-world startup success estimates and must not be used as investment advice.

## Key Features
- Indian startup financial inputs in INR (₹)
- Funding, industry, revenue, team size, and growth analysis
- Customer retention, monthly burn, and cash runway inputs
- Random Forest, Extra Trees, and Logistic Regression
- Soft Voting Ensemble classification
- Accuracy, F1-score, ROC-AUC, and Log Loss comparison
- Startup success probability output
- Business indicator observations
- Random Forest feature importance visualization
- Synthetic training dataset preview

## Technology Stack
Python · Streamlit · Pandas · NumPy · Scikit-learn

## Project Structure
```text
Startup-Success-Predictor-India/
├── main.py
├── requirements.txt
└── README.md
```

## Run Locally
1. Clone your repository:
   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```
2. Open the project directory:
   ```bash
   cd YOUR_PROJECT_FOLDER
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Deploy on Streamlit Community Cloud
1. Upload `main.py`, `requirements.txt`, and `README.md` to the root of your GitHub repository.
2. Open https://share.streamlit.io/.
3. Connect your GitHub repository and select its branch.
4. Set **Main file path** to `main.py`.
5. Deploy.

## Currency Units
| Input | Unit |
|---|---|
| Total funding | ₹ Crore |
| Annual revenue | ₹ Crore |
| Monthly operating burn | ₹ Lakh |
| Estimated market size | ₹ Crore |

## Disclaimer
This is an educational and portfolio demonstration project. The generated dataset is synthetic and does not represent verified startup histories. Performance on synthetic data should not be interpreted as real-world predictive performance. Do not use the output to make investment, lending, or business decisions.
