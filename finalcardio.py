"""
Cardiotocography (CTG) Data Analysis & Classification

This script analyzes fetal heart rate (FHR) signals from cardiotocography data
to classify fetal states (Normal, Suspect, Pathologic). It performs data cleaning,
exploratory data analysis (EDA), preprocessing, and trains multiple machine learning
models for comparison.

Dataset: CTG.xls (The Cardiotocography Data Set from UCI Machine Learning Repository)
Target Variable: 'NSP' (1: Normal, 2: Suspect, 3: Pathologic)

"""

# --- 1. Import Libraries ---
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
sns.set(style="ticks", palette="hls")


# --- 2. Data Loading & Cleaning ---
def load_and_clean_data(file_path):
    """
    Loads data from Excel, drops unrelated columns, and removes missing values.
    """
    try:
        # Load the 'Raw Data' sheet
        data = pd.read_excel(file_path, sheet_name='Raw Data')
        
        # Drop metadata columns irrelevant to classification
        # FileName, SegFile, and Date are administrative, not physiological
        cols_to_drop = ['FileName', 'SegFile', 'Date']
        data_clean = data.drop(columns=cols_to_drop)
        
        # Remove any rows with missing values (NaN)
        final_data = data_clean.dropna()
        
        print(f"Data loaded successfully. Shape: {final_data.shape}")
        return final_data
    except FileNotFoundError:
        print("Error: 'CTG.xls' not found. Please ensure the file is in the working directory.")
        return None

# Load the dataset
# Ensure 'CTG.xls' is in the same folder as this script
df = load_and_clean_data('CTG.xls')

if df is not None:
    # --- 3. Exploratory Data Analysis (EDA) ---
    print("\n--- Class Distribution (NSP) ---")
    
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='NSP', data=df, palette='hls')
    
    # Calculate and display percentages on bars
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{100 * height / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=12, color='black')
    
    plt.title('Distribution of Fetal State (NSP)', fontsize=20)
    plt.xlabel('NSP Class (1=Normal, 2=Suspect, 3=Pathologic)')
    plt.ylabel('Count')
    plt.show()

    # --- 4. Feature Selection & Splitting ---
    # The first 36 columns are features (LB, AC, FM, etc.)
    # The 'NSP' column is the target variable
    X = df.iloc[:, 0:36] 
    y = df['NSP']

    # Split into Training (75%) and Testing (25%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)

    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # --- 5. Preprocessing (Standardization) ---
    # Scale features to have mean=0 and variance=1.
    # Crucial: Fit scaler ONLY on training data to prevent data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 6. Model Training & Evaluation ---
    
    # Dictionary of classifiers to evaluate
    classifiers = {
        'Logistic Regression': LogisticRegression(multi_class='auto', max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Support Vector Machine': SVC(kernel='linear', probability=True)
    }

    print("\n--- Model Performance Results ---")

    for name, model in classifiers.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Predict on test data
        y_pred = model.predict(X_test_scaled)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.2%}") # Format as percentage
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.grid(False) # Turn off grid lines for clearer matrix
        plt.show()

else:
    print("Script terminated due to data loading error.")