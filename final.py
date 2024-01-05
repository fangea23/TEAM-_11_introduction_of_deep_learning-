import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from xgboost import XGBClassifier
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Input
from keras.optimizers import gradient_descent_v2, Adam


### Clustering

# -- KMeans --  
class KMeansClustering:
    # input = n * m
    # output = n * (m + 1)
    def __init__(self, n_clusters_range=(2, 11)):
        self.n_clusters_range = n_clusters_range
        self.k_range = range(*n_clusters_range)
        self.selected_K = None
        self.kmeans_model = None
        self.silhouette_scores = None

    def fit_predict(self, data):
        self.distortions = []
        self.silhouette_scores = []

        for i in self.k_range:
            kmeans = KMeans(n_clusters=i).fit(data)
            self.silhouette_scores.append(silhouette_score(data, kmeans.predict(data)))  # 側影係數

        # 找出最大的側影係數來決定 K 值
        self.selected_K = self.silhouette_scores.index(max(self.silhouette_scores)) + self.n_clusters_range[0]

        # 重新建立 KMeans 模型並預測目標值
        self.kmeans_model = KMeans(n_clusters=self.selected_K).fit(data)
        
        # 直接在原始數據中添加預測標籤
        labeled_data = np.column_stack((data, self.kmeans_model.predict(data)))
        print(labeled_data)
        return labeled_data


# -- Hierachical_clustering --
def Hierachical_clustering(input):
    # input = n * m
    # output = n * (m + 1)
    # 測試用Input
    # X = np.random.randint(5, size = (100,2))
    silhouette_avg = []
    for i in range(2, 11):
        model = AgglomerativeClustering(n_clusters=i)
        silhouette_avg.append(silhouette_score(input, model.fit(input).labels_))
    # 用silhouette_avg決定clustering數量
    # plt.plot(range(2,len(input)), silhouette_avg)
    k = np.argmax(silhouette_avg)
    model = AgglomerativeClustering(n_clusters=k)
    model.fit(input)
    labels = model.labels_
    labels = np.expand_dims(labels, 1)
    # 把input和label合併，變成n * (m + 1)輸出
    output = np.concatenate((input, labels),1)

    return output

### Classification

# -- XGBoost --    

### n_estimators為迭代次數
class XGBoostClassifier:
     def __init__(self, n_estimators=100, learning_rate=0.3):
          self.model = XGBClassifier(n_estimators = n_estimators, learning_rate = learning_rate)
     def train_and_predict(self, X_train, y_train, X_test):
        #   features = data[:,:-1]
        #   labels = data[:,-1]
        #   le = LabelEncoder()
        #   X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        #   y_train = le.fit_transform(y_train)

          ### Model Training
          self.model.fit(X_train, y_train)

          ### Model Predicted
          predicted = self.model.predict(X_test)
          score = self.model.score(X_train, y_train)
          
          return predicted, score
     

# -- Random Forest --
def random_forest(X_train, y_train, X_test):
    # input = n * (m + 1), n * m
    # output = score of y_train classification
    # X_train size = n * (m + 1)，其中前2行為Train, 最後1行為Label
    model = RandomForestClassifier(n_estimators=120, criterion='gini', random_state=0)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    return predicted

# -- ANN --

def Classification(training_data, testing_data):
    
    """
    參數：
    - testing_data (np.ndarray): 2D numpy數組，包含測試數據。
    - training_data (np.ndarray): 2D numpy數組，包含訓練數據，其中包括cluster前的資料作為特徵和cluster預測的分群結果。

    返回：
    np.ndarray: 包含對測試數據的預測結果的1D numpy數組。
    """
    # 提取訓練數據的特徵（cluster前的資料）
    training_features = training_data[:, :-1]

    # 提取訓練數據的真實標籤（cluster預測的分群結果）
    training_labels = training_data[:, -1]
    training_labels = to_categorical(training_labels, 10)
    
    model = Sequential()
    model.add(Input(shape=(training_features.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(training_features, training_labels, epochs=50)

    predictions = model.predict(testing_data)
    predicted_classes = np.argmax(predictions, axis=1)

    return predicted_classes