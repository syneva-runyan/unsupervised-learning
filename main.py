from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import mixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPClassifier

WORLD_CUP_DATA_SET_NAME = "World cup prediction data"
df_worldcup = pd.read_csv ('./world-cup.csv')
data_worldcup = df_worldcup[['spi', 'opposing_spi', 'spi_offense', 'opposing_spi_offense', 'spi_defense', 'opposing_spi_defense', 'sixteen']]

HEART_FAILURE_DATA_SET_NAME = 'Heart failure prediction data'
df_heartfailure = pd.read_csv ('./heart-failure.csv')
data_heartfailure = df_heartfailure[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]

# create test and train data
def create_train_and_test(test_size, df, data):
    labels = df[df.columns[-1]]
    train, test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size, stratify=labels)
    return [train, test, labels_train, labels_test, test_size]

def run_algorithm_against_data(alg, algName, df, independent_data, dataName):
  data = create_train_and_test(0.1, df, independent_data)
  alg(data[0], data[2], dataName)


def k_means_comprehensive(data, labels, dataName):
  print("Running K Means")
  intertias = []
   # check out the inertia for various cluster sizes
  for i in range(2,30):
    intertia = k_means_clustering(data, labels, dataName, i, 0)
    intertias.append(intertia)
  plt.plot(intertias)
  plt.ylabel('Intertia values')
  plt.title(dataName + ": Cluster Size vs Inertia Value")
  plt.show()

  if dataName == WORLD_CUP_DATA_SET_NAME:
    # selected n_cluster
    n = 10
  else: 
    n = 6
  
  k_means_clustering(data, labels, dataName, n, 1)


def run_algorithm(alg, algName):
  print("World Cup Data")
  run_algorithm_against_data(alg, algName, df_worldcup, data_worldcup, WORLD_CUP_DATA_SET_NAME)
  print("-------------")
  print("Heart Failure")
  run_algorithm_against_data(alg, algName, df_heartfailure, data_heartfailure, HEART_FAILURE_DATA_SET_NAME)
  
def k_means_clustering(data, labels, dataName, n_clusters, plot):
    start = time.perf_counter()
    clustered_data = KMeans(n_clusters=n_clusters, algorithm="full").fit(data)
    end = time.perf_counter()
    cluster_labels = clustered_data.labels_
    labels = np.array(labels)
    
    if plot == 0:
      if n_clusters == 2:
        print("Cluster Centers")
        print(clustered_data.cluster_centers_)
      return clustered_data.inertia_
  
    print("Time to run:" + str(end-start))
    yes = []
    no = []
    for index in range(0, n_clusters):
      yes.append(0)
      no.append(0)

    for index in range(0, len(cluster_labels)):
      cluster = cluster_labels[index]
      label = labels[index]
      if label == 1:
        yes[cluster] = yes[cluster] + 1
      else:
        no[cluster] = no[cluster] + 1
    
    ind = np.arange(n_clusters) 
    plt.bar(ind, yes, width=0.8, label='label - 1', color='green', bottom=no)
    plt.bar(ind, no, width=0.8, label='label - 0', color='red')

    plt.ylabel("Labels")
    plt.xlabel("Cluster")
    plt.legend(loc="upper right")
    plt.title(dataName + ": Clusters and Labels")

    plt.show()

def expectation_maximization(data, labels, dataName):
  estimator = mixture.GaussianMixture(n_components=2, init_params='random')
  start = time.perf_counter()
  estimator.fit(data)
  end = time.perf_counter()
  time_to_fit = end-start

  prediction_prob = estimator.predict_proba(data)
  score = estimator.score(data)
  print(score)
  labels = np.array(labels)
  yes = []
  no = []
  for index in range(0, len(labels)):
      c_0 = prediction_prob[index][0]
      c_1 = prediction_prob[index][1]
      label = labels[index]
      if label == 1:
        yes.append([c_0, c_1])
      else:
        no.append([c_0, c_1])
  yes_df = pd.DataFrame(yes, columns=['Prob Belonging to Cluster 0', 'Prob Belonging to Cluster 1'])
  no_df = pd.DataFrame(no, columns=['Prob Belonging to Cluster 0', 'Prob Belonging to Cluster 1'])
  plot_prob_cluster(yes_df, no_df, 'Prob Belonging to Cluster 0', dataName)
  plot_prob_cluster(yes_df, no_df, 'Prob Belonging to Cluster 1', dataName)

def plot_prob_cluster(yes_df, no_df, column, dataName):
  fig, ax = plt.subplots()
  ind_y = np.arange(len(yes_df)) 
  ind_x = np.arange(len(no_df)) + len(ind_y)
  ax.scatter(ind_y, np.array(yes_df[column]), c="green", label="Label - 1", alpha=0.5, edgecolors='none')
  ax.scatter(ind_x, np.array(no_df[column]), c="red", label="Label - 0", alpha=0.5, edgecolors='none')

  plt.ylabel("Probability of belonging to "+ column)
  plt.xlabel("Items")
  plt.legend(loc="upper right")
  plt.title(dataName + ": " + column)
  plt.show()

def pca_comprehensive(data, labels, dataName):
  print("Running  PCA")
  max = len(data.columns) - 1
  ratios = []
  scaler = MinMaxScaler()
  data_rescaled = scaler.fit_transform(data)
  for i in range(1, max):
    evr = pca(data_rescaled, dataName, i, 0)
    sumation = np.sum(evr)
    ratios.append(sumation)
  plt.plot(ratios)
  plt.ylabel('Cumulative Explained Variance Ratio')
  plt.xticks(np.arange(0, max, step=1))
  plt.axhline(y=0.95, color='red')
  plt.title(dataName + ": n_components vs Cumulative Explained Variance Ratio")
  plt.show()

  if dataName == WORLD_CUP_DATA_SET_NAME:
    # selected n_components
    n = 4
  else: 
    n = 7

  pca(data_rescaled, dataName, n, 1, labels)


def pca(data, dataName, n_components, transform, labels=None):
  pca = PCA(n_components=n_components)
  estimator = pca.fit(data)
  if transform == 0:
    return estimator.explained_variance_ratio_

  reduced_dataset = estimator.transform(data)
  print("Re run K Means Clustering")
  k_means_comprehensive(reduced_dataset, labels, dataName)
  print("Re run Expectation Maximization")
  expectation_maximization(reduced_dataset, labels, dataName)

def ica(data, labels, dataName):
  print("Running ICA")

  if dataName == WORLD_CUP_DATA_SET_NAME:
    # selected n_components
    n = 4
  else: 
    n = 7

  scaler = MinMaxScaler()
  data_rescaled = scaler.fit_transform(data)
  ica = FastICA(n_components=n, max_iter=3200, tol=0.00035)
  estimator = ica.fit(data_rescaled)
  reduced_dataset = estimator.transform(data_rescaled)
  print("Re run K Means Clustering")
  k_means_comprehensive(reduced_dataset, labels, dataName)
  print("Re run Expectation Maximization")
  expectation_maximization(reduced_dataset, labels, dataName)

def randomized_projection(data, labels, dataName):
  if dataName == WORLD_CUP_DATA_SET_NAME:
    # selected n_components
    n = 4
  else: 
    n = 7
  
  times_to_transform = []
  datasets = []
  for i in range(0, 1000):
    rp = GaussianRandomProjection(n_components=n)
    estimator = rp.fit(data)
    start = time.perf_counter()
    reduced_dataset = estimator.transform(data)
    end = time.perf_counter()
    datasets.append(reduced_dataset)
    times_to_transform.append(end-start)
  
  np_transform_time = np.array(times_to_transform)
  np_transform_time = np.mean(np_transform_time)
  
  print("Avg Time to transform" + str(np_transform_time))
  # randomly select 3 datasets to run k means and expectation maximazation on
  samples = random.sample(datasets, 3)
  for sample in samples:
    print("Re run K Means Clustering")
    k_means_comprehensive(sample, labels, dataName)
    print("Re run Expectation Maximization")
    expectation_maximization(sample, labels, dataName)

def feature_agglomeration(data, labels, dataName):
  print("Feature Agglomeration")
  if dataName == WORLD_CUP_DATA_SET_NAME:
    # selected n_components
    n = 4
  else: 
    n = 7
  
  fa = FeatureAgglomeration(n_clusters=2).fit(data)
  times_to_transform = []
  datasets = []
  for i in range(0, 1000):
    estimator = fa.fit(data)
    start = time.perf_counter()
    reduced_dataset = estimator.transform(data)
    end = time.perf_counter()
    datasets.append(reduced_dataset)
    times_to_transform.append(end-start)
    np_transform_time = np.array(times_to_transform)
    np_transform_time = np.mean(np_transform_time)
  
  print("Avg Time to transform" + str(np_transform_time))
  # randomly select 3 datasets to run k means and expectation maximazation on
  samples = random.sample(datasets, 3)
  for sample in samples:
    print("Re run K Means Clustering")
    k_means_comprehensive(sample, labels, dataName)
    print("Re run Expectation Maximization")
    expectation_maximization(sample, labels, dataName)

def getScore(classifier, train, test, labels_train, labels_test):
    start = time.perf_counter()
    classifier.fit(train, labels_train)
    end = time.perf_counter()
    time_to_train = end - start
    start = time.perf_counter()
    classifier.predict(test)
    end = time.perf_counter()
    time_to_predict = end - start
    score = classifier.score(test, labels_test)
    return time_to_train, time_to_predict, score

def neuralNet(train, test, labels_train, labels_test, data_name):
  classifier = MLPClassifier(solver='adam', activation='logistic')
  return getScore(classifier, train, test, labels_train, labels_test)

def getReducedRP():
  rp = GaussianRandomProjection(n_components=4)
  return rp.fit_transform(data_worldcup)

def getKMeansReduction():
  return KMeans(n_clusters=4, algorithm="full").fit_transform(data_worldcup)

def getEMReduction():
  emLabels = mixture.GaussianMixture(n_components=2, init_params='random').fit_predict(data_worldcup)
  np_labels = np.array(emLabels)
  data = np.expand_dims(np_labels, axis=0)
  return np.transpose(data)


def neuralNet_comprehensive(alg, its = 100):
  train_time_set = []
  predict_time_set = []
  score_set = []
  for i in range(0,its):
    print(i)
    reduced_dataset = alg()
    print(reduced_dataset.shape)
    data = create_train_and_test(0.1, df_worldcup, reduced_dataset)
    train_time, predict_time, score = neuralNet(data[0], data[1], data[2], data[3], WORLD_CUP_DATA_SET_NAME)
    train_time_set.append(train_time)
    predict_time_set.append(predict_time)
    score_set.append(score)
      
  # average of runs
  np_train = np.array(train_time_set)
  np_train = np.mean(np_train)
  print("time to train " + str(np_train))
  np_predict = np.array(predict_time_set)
  np_predict = np.mean(np_predict)
  print("time to predict " + str(np_predict))
  np_score = np.array(score_set)
  np_score = np.mean(np_score)
  print("score " + str(np_score))

run_algorithm(k_means_comprehensive, "K Means Clustering")
run_algorithm(expectation_maximization, "Expectation Maximization")
run_algorithm(pca_comprehensive, "PCA")
run_algorithm(ica, "ICA")
run_algorithm(randomized_projection, "Randomized Projection")
run_algorithm(feature_agglomeration, "Feature Agglomeraton")
neuralNet_comprehensive(getReducedRP)
neuralNet_comprehensive(getKMeansReduction)
neuralNet_comprehensive(getEMReduction)
