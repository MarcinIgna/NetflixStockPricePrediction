import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score 
from sklearn.preprocessing import PolynomialFeatures, scale, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.base import clone


data = pd.read_csv("NFLX.csv")
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].values.astype(np.int64) // 10 ** 9

# print(data.head(5))

X = data.drop('Close', axis=1)
y = data.Close

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Principal Component Analysis 
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)



# Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
predicted = lm.predict(X_test)


# Ridge Regression
rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)

# Lasso Regression
la = Lasso(alpha=0.1)
la.fit(X_train, y_train)

# Elastic Net Regression
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)


pca = PCA()
pca.fit(X_train)
X_train_hat = pca.transform(X_train)

N = 20
X_train_hat_PCA = X_train_hat[:, :N]  # Assign the transformed data to X_train_hat_PCA

# Now you can subset it
X_train_hat_PCA = X_train_hat_PCA[:, :N]

enet_pca = ElasticNet(tol=0.2, alpha=0.1, l1_ratio=0.1)
enet_pca.fit(X_train_hat_PCA, y_train)

# Functions
def get_R2_features(model, test=True): 
    # Evaluate R^2 for each feature
    features = X_train.columns.tolist()
    R_2_train = []
    R_2_test = []

    for feature in features:
        model_clone = clone(model)  # Create a copy of the model
        model_clone.fit(X_train[[feature]], y_train)
        R_2_test.append(model_clone.score(X_test[[feature]], y_test))
        R_2_train.append(model_clone.score(X_train[[feature]], y_train))

    # Plotting the results
    plt.bar(features, R_2_train, label="Train")
    plt.bar(features, R_2_test, label="Test")
    plt.xticks(rotation=90)
    plt.ylabel("$R^2$")
    plt.legend()
    plt.show()
    print("Training R^2 mean value {} Testing R^2 mean value {} ".format(str(np.mean(R_2_train)), str(np.mean(R_2_test))))
    print("Training R^2 max value {} Testing R^2 max value {} ".format(str(np.max(R_2_train)), str(np.max(R_2_test))))

print("Linear Regression")
get_R2_features(lm)
print("Ridge Regression")
get_R2_features(rr)
print("Lasso Regression")
get_R2_features(la)
print("Elastic Net Regression")
get_R2_features(enet)


def plot_coef(X, model, name=None):
    # Plotting coefficients
    plt.bar(X.columns, abs(model.coef_))
    plt.xticks(rotation=90)
    plt.ylabel("$coefficients$")
    plt.title(name)
    plt.show()
    print("R^2 on training data ", model.score(X_train, y_train))
    print("R^2 on testing data ", model.score(X_test, y_test))

def plot_dis(y, yhat):
    # Plotting distribution of actual vs fitted values
    plt.figure()
    ax1 = sns.kdeplot(y, color="r", label="Actual Value")
    sns.kdeplot(yhat, color="b", label="Fitted Values", ax=ax1)
    plt.legend()
    plt.title('Actual vs Fitted Values')
    plt.xlabel('Stock Price (in dollars)')
    plt.ylabel('Proportion of Stocks')
    plt.show()
    plt.close()

# After training your models, you can call these functions like this:

# For Linear Regression
predicted = lm.predict(X_test)
print("Linear Regression")
plot_coef(X_train, lm, "Linear Regression")
plot_dis(y_test, predicted)

# For Ridge Regression
predicted = rr.predict(X_test)
print("Ridge Regression")
plot_coef(X_train, rr, "Ridge Regression")
plot_dis(y_test, predicted)

# For Lasso Regression
predicted = la.predict(X_test)
print("Lasso Regression")
plot_coef(X_train, la, "Lasso Regression")
plot_dis(y_test, predicted)

# For Elastic Net Regression
predicted = enet.predict(X_test)
print("Elastic Net Regression")
plot_coef(X_train, enet, "Elastic Net Regression")
plot_dis(y_test, predicted)