# %%
import matplotlib.pyplot as plt 
import pandas as pd  
import numpy as np  
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# %%
df = pd.read_csv('https://drive.google.com/uc?id=1kCozLy5zNmPkeC6E47cQziPpjWpm5XZL')

# %% [markdown]
# ### Preprocessing

# %%
df.isnull().sum()

# %%
#Removing duplicate values
df = df.drop_duplicates()

# %%
# Removing the output variables and No 
removing_columns = ['Concrete-compressive-strength']

# %%
X = df[df.columns.difference(removing_columns)]

# %%
correlation_matrix = X.corr()
heatmap = sns.heatmap(data = correlation_matrix)
heatmap.set_title('Heatmap of correlation features', fontdict={'fontsize':20},pad=12)

# %%
#Removing the features which have more than 95% correlation
def columns_remove(corr_matrix, df):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return to_drop

# %%
drop = columns_remove(correlation_matrix,X)

# %%
# the drop cloumns is empty since there are no highly correlated features
drop

# %%
#Output Variable
Y = df.loc[:, df.columns == 'Concrete-compressive-strength']

# %% [markdown]
# ### Normalization

# %%
standard_scalar = StandardScaler()
X = pd.DataFrame(standard_scalar.fit(X).fit_transform(X))

# %% [markdown]
# ### Splitting the training and testing sets

# %%
# Splitting the data into training and testing with 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 5)

# %%
Y_train=Y_train.values.reshape(Y_train.shape[0],)
Y_test=Y_test.values.reshape(Y_test.shape[0],)


# %% [markdown]
# ### Linear regression library

# %%
#Linear Regression using ML libraries
#training the model using the LinearRegression  ML library
model = LinearRegression()
model.fit(X_train, Y_train, )

# %%
predict_Y_train = model.predict(X_train)
rmse_train = (np.sqrt(mean_squared_error(Y_train, predict_Y_train)))
r2_train = r2_score(Y_train, predict_Y_train)

# %%
#The performance of the model for training set
print('RMSE for training set is {}'.format(rmse_train))
print('R2 score for training set is {}'.format(r2_train))

# %%
predict_Y_test = model.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(Y_test, predict_Y_test)))
r2_test = r2_score(Y_test, predict_Y_test)

# %%
#The performance of the model for testing set
print('RMSE for testing set is {}'.format(rmse_test))
print('R2 score for testing set is {}'.format(r2_test))

# %%
# Scatter plot for training data
import seaborn as sns
sns.regplot(x= Y_train, y = predict_Y_train, color = 'red')
plt.title('Regression line with train data')


# %%
# Scatter plot for training data
sns.regplot(x = Y_test, y = predict_Y_test, color = 'blue')
plt.title('Regression line with test data')

# %%



