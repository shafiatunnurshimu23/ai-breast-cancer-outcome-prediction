# Predictive Modeling for Cancer Patient Survival

This repository contains the complete data analysis pipeline and proof-of-concept validation for the AI module of the BioBAT, a bio-inspired microrobotic system for targeted cancer therapy. This project demonstrates an end-to-end data analysis workflow, from raw clinical data to predictive modeling and interpretable insights, showcasing skills relevant for data analysis, data science, and biomedical informatics roles.

The full report detailing the project's background, design, and findings can be found [summary of analysis.docx](https://github.com/user-attachments/files/20903528/summary.of.analysis.docx) or in the repository.

 <!-- It's highly recommended to add a key image like the comparison table or ROC curve here -->

---

## 📂 Project Overview

The core of this project was to validate the feasibility of an AI-driven control system for a conceptual microrobot. To achieve this, a comprehensive analysis was performed on a real-world clinical dataset to predict patient survival outcomes. This served as a proof-of-concept, demonstrating that data-driven personalization in advanced therapeutics is not only plausible but practically achievable.

The analysis pipeline involved several key stages:
1.  **Data Cleaning and Preprocessing:** Handling missing values and preparing the dataset for analysis.
2.  **Feature Engineering:** Creating new, impactful features from existing data to enhance model performance.
3.  **Exploratory Data Analysis (EDA):** Identifying underlying trends, correlations, and distributions within the data.
4.  **Predictive Modeling:** Building, tuning, and evaluating multiple machine learning models to classify patient outcomes.
5.  **Model Interpretation (XAI):** Using SHAP to understand the "why" behind the model's predictions.
6.  **Statistical Survival Analysis:** Moving beyond classification to model time-to-event outcomes.

---

## 🛠️ Tech Stack & Libraries

This analysis was conducted in Python, leveraging the following core data science libraries:

- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn` (for Logistic Regression, Random Forest, GridSearchCV), `xgboost`
- **Imbalanced Data Handling:** `imbalanced-learn` (for SMOTE)
- **Model Interpretability:** `shap`
- **Survival Analysis:** `lifelines`

---

## 📊 Data Analysis Workflow & Key Findings

The primary analytical work is contained within the `analysis.ipynb` Jupyter Notebook.

### 1. Data Cleaning and Feature Engineering
- The initial `BRCA.csv` dataset was loaded and cleaned of null values in the target variable (`Patient_Status`).
- A crucial new feature, **`Survival_Time_Days`**, was engineered by calculating the delta between `Date_of_Surgery` and `Date_of_Last_Visit`. This transformed the dataset, enabling time-to-event survival analysis.
- Categorical features (e.g., `Tumour_Stage`, `Histology`) were converted to numerical format using `LabelEncoder`.

### 2. Exploratory Data Analysis (EDA)
- A **correlation heatmap** revealed weak inter-correlations among protein biomarkers, suggesting their value as independent predictors.
- **Distribution plots** stratified by patient status showed that `Protein4` expression was noticeably higher in the 'Dead' cohort, identifying it as a key feature of interest.
- A **pairplot** provided a comprehensive visual summary of feature relationships and confirmed the distinct clustering of patient outcomes based on key variables.

![output](https://github.com/user-attachments/assets/abfa323d-8054-4f76-858e-d023a7c8fda1)

 <!-- Add your pairplot here -->

### 3. Predictive Modeling & Performance
- The significant **class imbalance** in the dataset (191 'Alive' vs. 49 'Dead' in the training set) was addressed using the **SMOTE** technique to prevent model bias.
- Three classifiers were trained and compared: Logistic Regression, Random Forest, and a tuned **XGBoost** model.
- The XGBoost model demonstrated superior predictive performance, achieving an **Area Under the Curve (AUC) of 0.871** on the held-out test set, validating the model's ability to effectively distinguish between patient outcomes.
![ROC_Curves_Comparison](https://github.com/user-attachments/assets/0d45b221-5e60-43bd-aa67-fb364ec0932b)
 <!-- Add your ROC curve plot here -->

### 4. Model Interpretability & Insights (XAI)
- **SHAP (SHapley Additive exPlanations)** was used to interpret the "black box" XGBoost model.
- The analysis identified **`Survival_Time_Days`**, **`Protein4`**, and **`Age`** as the top three most influential features in predicting patient survival. This provides a data-driven basis for identifying key clinical biomarkers.
![SHAP_Bar_Plot](https://github.com/user-attachments/assets/15915199-0f59-42f9-bf4c-6a0633a47507)
![SHAP_Beeswarm_Plot](https://github.com/user-attachments/assets/23f7caa2-25e1-4aa1-bbb3-f0769a8a4d57)

 <!-- Add your SHAP bar plot here -->

### 5. Survival Analysis
- **Kaplan-Meier survival curves** showed a statistically significant difference (p < 0.001) in survival probability between patients of different tumor stages, confirming the clinical relevance of this feature.
- A **Cox Proportional-Hazards model** was fitted to quantify the impact of each feature on patient survival. The model identified `Protein4` as a borderline significant risk factor (p=0.05), with a **Hazard Ratio of 1.41**, indicating that a one-unit increase in its expression corresponds to a 41% increase in the risk of mortality.

---

## 🚀 How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap lifelines jupyter
    ```

3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

4.  **Open and run the `analysis.ipynb` notebook** to replicate the entire analysis workflow. Ensure the `BRCA.csv` file is in the same directory.

---

## 📈 Conclusion & Impact

This project successfully demonstrates a comprehensive data analysis workflow that validates the core AI component of a next-generation therapeutic system. It showcases the ability to derive actionable, interpretable insights from complex clinical data, establishing a strong, evidence-based foundation for future development in personalized medicine and intelligent robotics.

♻️ overall workFlow
![workflow_validation](https://github.com/user-attachments/assets/40a8ba6a-17f2-4e39-b289-0be773216a95)

