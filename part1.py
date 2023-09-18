import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


config = {
  'dataset': 'https://drive.google.com/uc?id=1kCozLy5zNmPkeC6E47cQziPpjWpm5XZL',
  'train-fraction': 0.8, 
  'epochs': 2000,
  'lr': 0.1,
  'logging': False,
  'correlation-threshold': 0.80,
  'actual-value': 'Concrete-compressive-strength',
  'convergence-threshold': 0.0005,
  'display-plots': False,
}



class LinearRegression:
  def __init__(self, dataset: str, actual_value: str):
    self.dataset = dataset
    self.actual_value = actual_value
    self.data = None
    self.scaled_data = None
    self.train_data = None
    self.test_data = None
    self.weights = None
    self.features_len = None
    self.train_x = None
    self.train_Y = None
    self.test_x = None
    self.test_Y = None
    self.log = []
    self.feature_stats = {}
    self.displayPlots = None
  
  def load_data(self):
    self.data = pd.read_csv(self.dataset)
    self.features_len = self.data.shape[1]
    if self.displayPlots:
      fig, axes = plt.subplots(3, 3, figsize=(8, 8))
      for i in range(3):
        for j in range(3):
            # Compute the index for the current subplot
            index = i * 3 + j
            # Create a scatter plot in the current subplot
            axes[i, j].scatter(self.data.iloc[:,index], self.data.loc[:, self.actual_value], label=f'Subplot {index}')
            axes[i, j].set_title(f'Subplot {self.data.columns[index]}')
            axes[i, j].legend()

      plt.tight_layout()
      plt.show()

      for col in self.data.columns:
        plt.figure(figsize=(8,8))
        ax=sns.distplot(self.data[col], fit=norm)
        ax.axvline(self.data[col].mean(), color='magenta', linestyle='dashed', linewidth=2)
        ax.axvline(self.data[col].median(), color='teal', linestyle='dashed', linewidth=2)
        ax.set_title(f'skewness of {col} : {self.data[col].skew()}')
        plt.show()

    
    
  
  def filter_duplicates(self):
    duplicates = self.data.duplicated()
    if len(duplicates):
      print("found duplicates\n")
      print(self.data[duplicates], '\n')
      print('removing duplicate rows \n')
      self.data = self.data.drop_duplicates()
      print('duplicate rows removed\n')
    else:
      print("no duplicates found")

  def feature_selection(self, threshold):
    correlation_matrix = (self.data.drop(columns=[self.actual_value])).corr()
    print("correlation matrix within the features:\n")
    print(correlation_matrix)
    print("\n")
    high_corr_pairs = [(self.data.columns[i], self.data.columns[j]) for i in range(len(correlation_matrix.columns)) 
                   for j in range(i+1, len(correlation_matrix.columns)) 
                   if abs(correlation_matrix.iloc[i, j]) > threshold]
    print("column pairs with high correlation:\n")
    print(high_corr_pairs)
    print("\n")
    columns_to_drop = set()
    columns_remain = set()
    for i in high_corr_pairs:
      corr0 = self.data[i[0]].corr(self.data[self.actual_value])
      corr1 = self.data[i[1]].corr(self.data[self.actual_value])
      print(f'correlation of {i[0]} with {self.actual_value} is {corr0}\n')
      print(f'correlation of {i[1]} with {self.actual_value} is {corr1}\n')
      if abs(corr0) > abs(corr1):
        if i[0] not in columns_remain:
          columns_remain.add(i[0])
        if i[1] not in columns_to_drop:
          columns_to_drop.add(i[1])
      else:
        if i[1] not in columns_remain:
          columns_remain.add(i[1])
        if i[0] not in columns_to_drop:
          columns_to_drop.add(i[0])
    print('columns to remove \n')
    print(columns_to_drop, '\n')
    print('columns to remain\n')
    print(columns_remain, '\n\n')
    print(f'removing features: {",".join(list(columns_to_drop))}\n')
    self.data = self.data.drop(columns=list(columns_to_drop))
    self.features_len = len(self.data.columns)
    if self.displayPlots:
      plt.figure(figsize=(10, 5))
      sns.heatmap(correlation_matrix, vmin=-1, cmap='coolwarm', annot=True)
      plt.show()

  def normalize_features(self):
    # using min max scaling
    self.scaled_data = self.data
    for column in self.data.columns:
      if column != self.actual_value:
        min = self.data[column].min()
        max = self.data[column].max()
        self.scaled_data[column] = (self.data[column] - min)/(max-min)
        self.feature_stats[column]={
          'min': min,
          'max': max
        }
    print('normalized features')
    return

  def preprocess_data(self, correlation_threshold):
    # checking for null values
    if self.data.isnull().sum().sum():
      print('Null values are present')
    else:
      print('No null values are present')
    
    # filtering duplicate rows
    self.filter_duplicates()

    # filtering features based on the correlation between them
    self.feature_selection(correlation_threshold)

    # normalizing the filtered features
    self.normalize_features()
    
  def split_data(self, split: float):
    self.train_data = self.scaled_data.sample(frac=split)
    self.test_data = self.scaled_data.drop(self.train_data.index)

  def train_split(self):
    self.train_x = self.train_data.drop(columns=[self.actual_value], axis=1)
    self.train_x.insert(0, 'constant', 1)
    self.train_Y = np.array(self.train_data[self.actual_value]).reshape(self.train_x.shape[0], 1)

  def test_split(self):
    self.test_x = self.test_data.drop(columns=[self.actual_value], axis=1)
    self.test_x.insert(0, 'constant', 1)
    self.test_Y = np.array(self.test_data[self.actual_value]).reshape(self.test_x.shape[0], 1)

  def initialize_weights(self):
    self.weights = np.random.rand(self.features_len, 1)

  def R_squared_stat(self, actual, pred):
    mean_y = actual.mean()
    RSS = ((actual - pred)**2).sum()
    TSS = ((actual - mean_y)**2).sum()
    return 1 - (RSS/TSS)
    
  def MSE(self, actual, pred):
    return (((actual - pred)**2).sum())/actual.shape[0]

  def RMSE(self, mse):
    return np.sqrt(mse)

  def testLoss(self):
    return self.MSE(self.test_Y, self.test_x.dot(self.weights))

  def logData(self):
    fields = ['epoch', 'trainingLoss', 'testLoss', 'epochs', 'lr', 'training_r2', 'testing_r2', 'training_rmse', 'testing_rmse']
    fields = fields + ['W'+str(i) for i in range(self.features_len)]
    logFile = open('part1-logfile.csv', 'a')
    logFile.truncate(0)
    writer = csv.DictWriter(logFile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(self.log)
    logFile.close()

  def gradient_descent(self, train_T, lr, conv_threshold):
    pred_output = self.train_x.dot(self.weights)
    difference = (lr/self.train_x.shape[0])*(train_T.dot(pred_output - self.train_Y))
    if np.max(np.abs(difference)) > conv_threshold:
      self.weights -= difference
      return { 'converged': False, 'out': pred_output }
    else:
      return { 'converged': True, 'out': pred_output }

  def train(self, epochs: int, lr: float, conv_threshold: float):
    train_trans = np.array(self.train_x).T
    for epoch in range(epochs):
      result = self.gradient_descent(train_trans, lr, conv_threshold)
      if not result['converged']:
        trainingLoss = self.MSE(self.train_Y, result['out'])
        testLoss = self.testLoss()
        training_r2 = self.R_squared_stat(self.train_Y, result['out'])
        testing_r2 = self.R_squared_stat(self.test_Y, self.test_x.dot(self.weights))
        training_rmse = self.RMSE(trainingLoss)
        testing_rmse = self.RMSE(testLoss)
        logData = {
          'epoch': epoch,
          'trainingLoss': trainingLoss.iloc[0],
          'testLoss': testLoss.iloc[0],
          'lr': lr,
          'epochs': epochs,
          'training_r2': training_r2.iloc[0],
          'testing_r2': testing_r2.iloc[0],
          'training_rmse': training_rmse.iloc[0],
          'testing_rmse': testing_rmse.iloc[0]
        }
        for i in range(len(self.weights)):
          logData['W'+str(i)] = self.weights[i][0]
        self.log.append(logData)
      else:
        print('convergence threshold value reached. stopping the training.\n')
        break
        
    print(self.log[-1])
    if self.displayPlots:
      sns.regplot(x = self.test_Y, y = self.test_x.dot(self.weights), color='green').set_title('Regression line with test data')
      plt.show()
      sns.regplot(x = self.train_Y, y = self.train_x.dot(self.weights), color='green').set_title('Regression line with train data')
      plt.show()
  
  def predict(self, features):
    input = [1]
    for key in features.keys():
      input.append((features[key] - self.feature_stats[key]['min']) / (self.feature_stats[key]['max'] - self.feature_stats[key]['min']))
    input = np.array(input)
    print(input)
    return input.dot(self.weights)

  def fit(self, split, epochs, lr, threshold, conv_threshold, logging:bool = False, displayPlots: bool = False):
    self.displayPlots = displayPlots
    self.load_data()
    self.preprocess_data(threshold)
    self.split_data(split)
    self.train_split()
    self.test_split()
    self.initialize_weights()
    self.train(epochs, lr, conv_threshold)
    if logging:
      self.logData()
    if self.displayPlots and logging:
      logDf = pd.read_csv('./part1-logfile.csv')
      # 'MSE plot Training'
      plt.plot(logDf['epoch'], logDf['trainingLoss'])
      plt.title('MSE training data vs epoch')
      plt.show()
      # 'MSE plot Testing'
      plt.plot(logDf['epoch'], logDf['testLoss'])
      plt.title('MSE testing data vs epoch')
      plt.show()
      # 'R2 plot Training'
      plt.plot(logDf['epoch'], logDf['training_r2'])
      plt.title('Training R2 score vs epoch')
      plt.show()
      # 'R2 plot Testing'
      plt.plot(logDf['epoch'], logDf['testing_r2'])
      plt.title('Testing R2 score vs epoch')
      plt.show()
      # 'RMSE plot Training'
      plt.plot(logDf['epoch'], logDf['training_rmse'])
      plt.title('Training RMSE vs epoch')
      plt.show()
      # 'RMSE plot Testing'
      plt.plot(logDf['epoch'], logDf['testing_rmse'])
      plt.title('Testing RMSE vs epoch')
      plt.show()
      for i in range(self.features_len):
        plt.plot(logDf['epoch'], logDf['W'+str(i)])
        plt.title(f'W{i} vs epoch')
        plt.show()
      


model = LinearRegression(config['dataset'], config['actual-value'])
model.fit(config['train-fraction'], config['epochs'], config['lr'], config['correlation-threshold'],config['convergence-threshold'], config['logging'], config['display-plots'])
# 139.6 ,209.4 ,0.0 ,192.0 ,0.0 ,1047.0 ,806.9 ,28 ,28.24 

result = model.predict({
  'Cement': 139.6,
  'Blast Furnace Slag': 209.4,
  'Fly Ash': 0.0,
  'Water': 192.0,
  'Superplasticizer': 0.0,
  'Coarse Aggregate':1047.0,
  'Fine Aggregate': 806.9,
  'Age': 28,
  # 'Concrete-compressive-strength': 28.24
})

print(result)