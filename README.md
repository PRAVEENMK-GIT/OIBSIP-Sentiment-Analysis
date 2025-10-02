# Sentiment Analysis Project

This project demonstrates a complete workflow for sentiment analysis using machine learning and NLP techniques on Twitter data.

## Project Steps
1. **Load and Explore the Dataset**
   - Load the CSV data and visualize sentiment distribution.
2. **Preprocess Text Data**
   - Clean, tokenize, and lemmatize text for better feature extraction.
3. **Feature Extraction**
   - Convert text to numerical features using TF-IDF vectorization.
4. **Train Sentiment Classification Models**
   - Train Naive Bayes and SVM models to classify sentiment.
5. **Evaluate Model Performance**
   - Assess accuracy, precision, recall, F1-score, and confusion matrix.
6. **Visualize Results**
   - Display sentiment distribution and word cloud for positive sentiment.

## Key Findings & Insights
- The dataset contains negative, neutral, and positive sentiments, visualized for better understanding.
- Text preprocessing and feature engineering are crucial for model performance.
- SVM generally outperforms Naive Bayes for text classification.
- Confusion matrix reveals misclassification trends and data imbalance.
- Word cloud highlights frequent words in positive tweets, offering insight into public opinion.

## Usage
Open `Sentiment_Analysis.ipynb` and run the cells sequentially to reproduce the analysis and visualizations.

## Requirements
- Python 3
- pandas, matplotlib, seaborn, scikit-learn, nltk, wordcloud

Install missing packages using pip if needed:
```
pip install pandas matplotlib seaborn scikit-learn nltk wordcloud
```

## Data
- `Twitter_Data.csv`: Contains cleaned tweet text and sentiment labels.

## Author
Praveen MK
