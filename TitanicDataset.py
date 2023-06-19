import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i]=data[i]

        for i in range(self.max_iter):
            self.classifications={}

            for j in range(self.k):
                self.classifications[j] = []

            for featureset in X:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                #ON FIRST RUN DON'T DO THIS!!! ... BUT THEN ADD
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if original_centroid[0]!=0 and original_centroid[1]!=0:
                    if np.sum((current_centroid-original_centroid)/(original_centroid)*100)>self.tol:
                        print(np.sum((current_centroid-original_centroid)/original_centroid*100))
                        optimized = False
                else: continue

            if optimized:
                break
    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

df = pd.read_excel('titanic.xls')
df.drop(['name','body'],axis=1,inplace=True)
#df.convert_objects(convert_numeric=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(0,inplace=True)
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
#print(df.loc[[1,15]])


df.drop(['ticket','home.dest'], axis=1, inplace=True)
#df.drop(['boat'],axis=1, inplace=True)
print(df.head())
X = np.array(df.drop(['survived'],axis=1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
#clf = KMeans(n_clusters=2)
clf = K_Means()
clf.fit(X)

correct=0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1
print(correct/len(X))
