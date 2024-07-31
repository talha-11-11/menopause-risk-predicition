# eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(df):
    # Plot distribution of menopause status
    sns.countplot(df['menopause_status'])
    plt.show()

    # Pairplot of features
    sns.pairplot(df)
    plt.show()

    # Correlation heatmap
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()
