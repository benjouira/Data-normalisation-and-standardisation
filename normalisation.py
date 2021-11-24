# ******************************************************  Feature Scaling with Python
# Application de la normalisation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
 
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'
 
names= ['constructor','Model','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
 
dataset = pd.read_csv(url, names=names)
#dataset = pd.read_csv("machine.csv", names=names)
 
# MIN MAX SCALING
minmax_scale = MinMaxScaler().fit(dataset[['MYCT', 'MMAX']])
df_minmax = minmax_scale.transform(dataset[['MYCT', 'MMAX']])
 
#imprimer un retour à la ligne pour une meilleur clarete de lecture
print('\n********** Normalisation*********\n')
 
print('Moyenne apres le Min max Scaling :\nMYCT={:.2f}, MMAX={:.2f}'
.format(df_minmax[:,0].mean(), df_minmax[:,1].mean()))
 
print('\n')
 
print('Valeur minimale et maximale pour la feature MYCT apres min max scaling: \nMIN={:.2f}, MAX={:.2f}'
.format(df_minmax[:,0].min(), df_minmax[:,0].max()))
 
print('\n')
 
print('Valeur minimale et maximale pour la feature MMAX apres min max scaling : \nMIN={:.2f}, MAX={:.2f}'
.format(df_minmax[:,1].min(), df_minmax[:,1].max()))
