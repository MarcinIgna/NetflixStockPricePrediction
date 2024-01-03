# Stock Price Predictor

This program uses various regression models to predict the closing stock prices of a company. The models used include Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net Regression. The program also applies Principal Component Analysis (PCA) to the training data for further analysis.

## Installation Requirements

The program requires the following Python libraries:

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

You can install these libraries using pip and the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

## Usage

The program reads stock price data from a CSV file, trains the regression models on this data, and then uses the models to predict the closing stock prices for the test data. The performance of each model is evaluated using the R^2 score, and the coefficients of each model are plotted to understand the influence of each feature on the prediction.

---







link to the dataset download: https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction/download?datasetVersionNumber=1

link to the dataset: https://www.kaggle.com/datasets/jainilcoder/netflix-stock-price-prediction/data