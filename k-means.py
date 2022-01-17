#loading libraries
from curses import COLOR_BLACK, COLOR_BLUE, COLOR_RED
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

#loading iris dataset
iris = datasets.load_iris()

#displaying our dataset for getting best visualisation
#print(iris)
#print(iris.data)
#print(iris.feature_names)
#print(iris.target)
#print(iris.target_names)

#Storing data as Panda's DataFrame 
x=pd.DataFrame(iris.data)
# définir les noms de colonnes
x.columns=['Sepal_Length','Sepal_width','Petal_Length','Petal_width']
y=pd.DataFrame(iris.target)
y.columns = ['Targets']

st.text('Displaying data..., how many?')
#dataframe head number from slider
headVal = st.slider('Head:', min_value = 2, max_value = x.shape[0])
st.dataframe(x.head(headVal))

#cluster number from slider

n_clusters = st.slider('Cluster:', min_value = 2, max_value = 12)

#Cluster K-means
model = KMeans(n_clusters)
#adapter le modèle de données
model.fit(x)

#st.write("'Loading data...", model.labels_)

#print(model.labels_)

colorList = np.array(['Red','green','blue','yellow','purple','black','grey','orange','blue','Red','yellow','blue'])

fig, ax = plt.subplots()
plt.scatter(x.Petal_Length, x.Petal_width, c=colorList[y.Targets], s=40)
plt.scatter(x.Petal_Length, x.Petal_width, c=colorList[model.labels_], s=40)

st.pyplot(fig)
