#This program shows the accuracy via R-squared and RMSE score for a house sale dataset on the following models
#* Multiple Linear Regression
#* Ridge
#* Lasso
#* Artificial Neural Network

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error, r2_score

#Reads in house sale data
df = pd.read_csv("house_sale_data.csv")

df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('float64')

#Selected features
X = df[["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
               "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
               "GrLivArea", "BsmtFullBath", "BsmtHalfBath"]].copy().to_numpy()

y = df["SalePrice"].copy().to_numpy()

#Subtract avg and divide my std. deviation
X -= np.average(X, axis=0)
X /= np.std(X, axis=0)
y -= np.average(y)
y /= np.std(y)

#Split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

#Multiple Linear Regression********************************************************************************************
lin = LinearRegression().fit(X_train, y_train)
print("***sklearn Linear Regression***")
print(f"R Squared: {lin.score(X_test, y_test)}")
print(f"RMSE: {np.sqrt(np.average((y-lin.predict(X))**2.0))}")
print()

#Ridge Regression******************************************************************************************************
parameters = {"alpha": np.linspace(0.01, 1, num=10)}

rid = Ridge(alpha=1.0).fit(X_train, y_train)
print("***sklearn Ridge Regression***")

#Grid search uses k-fold validation to find the optimal alpha
grid_search = GridSearchCV(rid, param_grid=parameters, cv=5, scoring="r2") #Score with r2 like other regressions
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_alpha', 'mean_test_score', 'rank_test_score']])

#nsmallest: bring best alpha to top row, iloc[0]: isolate top row, ['param_alpha']: get val from this col
optAlpha = round((score_df.nsmallest(1, 'rank_test_score')).iloc[0]['param_alpha'], 2)
rid.alpha = optAlpha

print(f"----Final Testing W/ Optimal Alpha: {rid.alpha}----")
print(f"R Squared: {rid.score(X_test, y_test)}")
print(f"RMSE: {np.sqrt(np.average((rid.predict(X_test) - y_test) ** 2.0))}")
print()

#Lasso Regression******************************************************************************************************
parameters = {"alpha": np.linspace(0.01, 0.1, num=10)}

las = Lasso(alpha=0.2).fit(X_train, y_train)
print("***sklearn Lasso Regression***")

#Grid search uses k-fold validation to find the optimal alpha
grid_search = GridSearchCV(las, param_grid=parameters, cv=5, scoring="r2") #Score with r2 like other regressions
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_alpha', 'mean_test_score', 'rank_test_score']])

#nsmallest: bring best alpha to top row, iloc[0]: isolate top row, ['param_alpha']: get val from this col
optAlpha = round((score_df.nsmallest(1, 'rank_test_score')).iloc[0]['param_alpha'], 2)
las.alpha = optAlpha

print(f"----Final Testing W/ Optimal Alpha: {las.alpha}----")
print(f"R Squared: {las.score(X_test, y_test)}")
print(f"RMSE: {np.sqrt(np.average((las.predict(X_test) - y_test) ** 2.0))}")
print()

#ANN*******************************************************************************************************************

class HouseData(Dataset):
    def __init__(self):
        df = pd.read_csv("house_sale_data.csv")

        df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('float32')

        # Selected features
        X = df[["MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
                "GrLivArea", "BsmtFullBath", "BsmtHalfBath"]].copy().to_numpy()

        y = df["SalePrice"].copy().to_numpy()

        # Subtract avg and divide my std. deviation
        X -= np.average(X, axis=0)
        X /= np.std(X, axis=0)
        y -= np.average(y)
        y /= np.std(y)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # Determine the length of the dataset
        self.len = self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

    def to_numpy(self):
        return np.array(self.X.view(-1)), np.array(self.y)

class HouseFit(nn.Module):
    def __init__(self):
        super(HouseFit, self).__init__()

        self.in_to_h1 = nn.Linear(16, 4)
        self.h1_to_out = nn.Linear(4, 1)

    def forward(self, x):
        x = F.sigmoid(self.in_to_h1(x))
        x = self.h1_to_out(x)
        return x

def trainNN(epochs=100, batch_size=16, lr=0.001, epoch_display=50):
    cd = HouseData()

    # Create a data loader that shuffles each time and allows for the last batch to be smaller
    # than the rest of the epoch if the batch size doesn't divide the training set size
    house_loader = DataLoader(cd, batch_size=batch_size, drop_last=False, shuffle=True)

    # Create my neural network
    house_network = HouseFit()

    # Mean square error (RSS)
    mse_loss = nn.MSELoss(reduction='sum')

    # Select the optimizer
    optimizer = torch.optim.Adam(house_network.parameters(), lr=lr)

    running_loss = 0.0

    for epoch in range(epochs):
        for _, data in enumerate(house_loader, 0):
            x, y = data

            optimizer.zero_grad()  # resets gradients to zero

            output = house_network(x)  # evaluate the neural network on x

            loss = mse_loss(output.view(-1), y)  # compare to the actual label value

            loss.backward()  # perform back propagation

            optimizer.step()  # perform gradient descent with an Adam optimizer

            running_loss += loss.item()  # update the total loss

        # every epoch_display epochs give the mean square error since the last update
        # this is averaged over multiple epochs
        if epoch % epoch_display == epoch_display - 1:
            print(
                f"Epoch {epoch + 1} / {epochs} Average loss: {running_loss / (len(house_loader) * epoch_display):.6f}")
            running_loss = 0.0
    return house_network, cd


#Evaluates the accuracy of the model
def evaluate_model(model, dataset):
    model.eval()
    with torch.no_grad():
        X = dataset.X
        y_true = dataset.y
        y_pred = model(X).view(-1)

        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()

        rmse = mean_squared_error(y_true_np, y_pred_np, squared=False)
        r2 = r2_score(y_true_np, y_pred_np)

        print(f"\nEvaluation on Full Dataset:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

model, dataset = trainNN(epochs=400)
evaluate_model(model, dataset)