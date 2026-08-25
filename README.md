# SkillSync — Resume Analysis & Job-Fit Prediction

SkillSync is a machine learning project designed to analyze resumes, classify candidates into job categories, evaluate resume quality, and predict how well a resume matches a job description.

## Features

- Resume category classification
- Resume quality clustering
- Resume–job description matching
- Job-fit prediction
- LinkedIn job-posting analysis
- Resume feature extraction
- Model evaluation using Accuracy, Macro-F1, RMSE, MAE, and Silhouette Score

## Machine Learning Models

### Classification
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Multi-Layer Perceptron (MLP)

### Clustering
- K-Means
- DBSCAN
- PCA for visualization

### Resume–Job Matching
- TF-IDF
- Cosine Similarity
- Logistic Regression
- Probability calibration

## Dataset

The project uses multiple datasets for different tasks.

### Resume Dataset

The resume classification and clustering components use a dataset containing:

- **2,484 resumes**
- **24 professional/job categories**

The raw `Resume.csv` dataset is not included in the repository.

### Resume–Job Fit Dataset

The job-fit component uses the `cnamuangtoun/resume-job-description-fit` dataset from Hugging Face.

The dataset contains resume and job-description pairs with fit labels such as:

- No Fit
- Potential Fit
- Good Fit

## Project Structure

```text
ML_Project/
│
├── Person 1.ipynb
├── Person3_day3.ipynb
├── Person3_week2.ipynb
├── week1.ipynb
│
├── note1.ipynb
├── note2.ipynb
├── note3.ipynb
├── note4.ipynb
│
├── eval_utils.py
│
├── classification_results.csv
├── clustering_results.csv
├── case_study_results.csv
├── filtered_linkedin_postings.csv
└── requirements.csv
