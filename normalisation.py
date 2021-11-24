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
# ********************************

# Application de la standardisation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
 
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data'
 
names= ['constructor','Model','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP']
 
 
dataset = pd.read_csv(url, names=names)
 
# Z-Score standardisation
 
std_scaler = MinMaxScaler().fit(dataset[['MYCT', 'MMAX']])
df_std = std_scaler.transform(dataset[['MYCT', 'MMAX']])
 
print('\n********** Standardisation*********\n')
 
print('Moyenne et Ecart type apres la standardisation de la feature MYCT :\nMoyenne={:.2f}, Ecart Type={:.2f}'
.format(df_std[:,0].mean(), df_std[:,0].std()))
 
print('\n')
 
print('Moyenne et Ecart type apres la standardisation de la feature MYCT :\nMoyenne={:.2f}, Ecart Type={:.2f}'
.format(df_std[:,1].mean(), df_std[:,1].std()))
 
print('\n')
 
print('Valeur minimale et maximale pour la feature MYCT apres min max scaling: \nMIN={:.2f}, MAX={:.2f}'
.format(df_std[:,0].min(), df_std[:,0].max()))
print('\n')
 
print('Valeur minimal et maximal pour la feature MMAX apres min max scaling : \nMIN={:.2f}, MAX={:.2f}'
.format(df_std[:,1].min(), df_std[:,1].max()))
 
print('Affichage des valeurs apres scaling')
print(df_std)
