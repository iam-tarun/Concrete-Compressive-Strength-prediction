import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns

config = {
  'dataset': './dataset/winequality-white.csv',
  'train-fraction': 0.8, 
  'epochs': 110,
  'lr': 0.002,
  'logging': True,
  'correlation-threshold': 0.8,
  'actual-value': 'quality',
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
  
  def load_data(self):
    self.data = pd.read_csv(self.dataset, sep=';')
    # print(self.data.head)
    # print(self.data.shape[1])
    self.features_len = self.data.shape[1]
    # self.weights = np.random.rand(1, self.features_len)
    # print("initial random values are ",self.weights)
  
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
    correlation_matrix = (self.data.drop(columns=[self.actual_value])).corr(method='spearman')
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

  def normalize_features(self):
    # using min max scaling
    self.scaled_data = self.data
    for column in self.data.columns:
      if column != self.actual_value:
        min = self.data[column].min()
        max = self.data[column].max()
        self.scaled_data[column] = (self.data[column] - min)/(max-min)
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
    self.train_x = self.train_data.drop(columns=["quality"], axis=1)
    self.train_x.insert(0, 'constant', 1)
    self.train_Y = np.array(self.train_data['quality']).reshape(self.train_x.shape[0], 1)

    # print(self.train_Y)
    # print(self.train_x)

  def test_split(self):
    self.test_x = self.test_data.drop(columns=['quality'], axis=1)
    self.test_x.insert(0, 'constant', 1)
    self.test_Y = np.array(self.test_data['quality']).reshape(self.test_x.shape[0], 1)

  def initialize_weights(self):
    self.weights = np.random.rand(self.features_len, 1)
    # print(self.weights)
    # self.weights = np.tile(self.weights, (self.features_len, self.train_x.shape[0]))
    # print(self.weights)

  def MSE(self, actual, pred):
    return (((actual - pred)**2).sum())/actual.shape[0]

  def testLoss(self):
    return self.MSE(self.test_Y, self.test_x.dot(self.weights))

  def logData(self):
    fields = ['epoch', 'trainingLoss', 'testLoss', 'epochs', 'lr']
    logFile = open('part1-logfile.csv', 'a')
    writer = csv.DictWriter(logFile, fieldnames=fields)
    writer.writeheader()
    writer.writerows(self.log)
    logFile.close()

  def train(self, epochs: int, lr: float):
    train_trans = np.array(self.train_x).reshape(self.features_len, self.train_x.shape[0])
    for epoch in range(epochs):
      pred_output = self.train_x.dot(self.weights)
      self.weights = self.weights - (lr/self.train_x.shape[0])*(train_trans.dot(pred_output - self.train_Y))
      error = self.MSE(self.train_Y, pred_output)
      testLoss = self.testLoss()
      self.log.append({
        'epoch': epoch,
        'trainingLoss': float(error.iloc[0]),
        'testLoss': float(testLoss.iloc[0]),
        'lr': lr,
        'epochs': epochs
      })

  def predict(self, features):
    return features.dot(self.weights)

  def fit(self, split, epochs, lr, threshold, logging:bool = False):
    self.load_data()
    self.preprocess_data(threshold)
    self.split_data(split)
    self.train_split()
    self.test_split()
    self.initialize_weights()
    self.train(epochs, lr)
    if logging:
      self.logData()



    
  
  

    


model = LinearRegression(config['dataset'], config['actual-value'])
model.fit(config['train-fraction'], config['epochs'], config['lr'], config['correlation-threshold'], config['logging'])
