from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

WORLD_CUP_DATA_SET_NAME = "World cup prediction data"
df_worldcup = pd.read_csv ('./world-cup.csv')
data_worldcup = df_worldcup[['spi', 'opposing_spi', 'spi_offense', 'opposing_spi_offense', 'spi_defense', 'opposing_spi_defense', 'sixteen']]

HEART_FAILURE_DATA_SET_NAME = 'heart_failure_prediction_data'
df_heartfailure = pd.read_csv ('./heart-failure.csv')
data_heartfailure = df_heartfailure[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]

# create test and train data
def create_train_and_test(test_size, df, data):
    n = len(df.columns)
    labels = df[df.columns[-1]]
    train, test, labels_train, labels_test = train_test_split(data, labels, test_size=test_size, stratify=labels)
    return [train, test, labels_train, labels_test, test_size]

def run_algorithm_against_data(alg, algName, df, independent_data):
  data = create_train_and_test(0.1, df, independent_data)
  alg(data[0])

def run_algorithm(alg, algName):
  print("World Cup Data")
  run_algorithm_against_data(alg, algName, df_worldcup, data_worldcup)
  print("-------------")
  print("Heart Failure")
  run_algorithm_against_data(alg, algName, df_heartfailure, data_heartfailure)

# k means clustering
def k_means_clustering(train):
  kmeans = KMeans(n_clusters=2).fit(train)
  plotClusters(kmeans)


def plotClusters(cluster_data):
  fig = plt.figure(1, figsize=(4, 3))
  ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
  labels = cluster_data.labels_
  print(labels)

  # ax.scatter(X[:, 3], X[:, 0], X[:, 2],
  #             c=labels.astype(np.float), edgecolor='k')

  # ax.w_xaxis.set_ticklabels([])
  # ax.w_yaxis.set_ticklabels([])
  # ax.w_zaxis.set_ticklabels([])
  # ax.set_xlabel('Petal width')
  # ax.set_ylabel('Sepal length')
  # ax.set_zlabel('Petal length')
  # ax.set_title(titles[fignum - 1])
  # ax.dist = 12


run_algorithm(k_means_clustering, "K Means Clustering")