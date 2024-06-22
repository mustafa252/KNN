
# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
# split
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.datasets import load_wine

# load dataset
wine = load_wine(as_frame=True)
wine.keys()
data = wine.data
features = wine.feature_names
targets = wine.target
wine.target_names

# describtion
print(wine['DESCR'])


#####################################################################################
##################### csv file

Df = wine['frame']
Df
Df.info()
Df.isnull().sum()


##########################################################################################
################# data analysis

# quality + size
plt.rcParams['figure.figsize'] = [9,3]
plt.rcParams['figure.dpi'] = 300

# behaivor of data (classes, +Reg, -Reg)
sns.pairplot(Df, vars=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium'],
             kind='reg', hue='target')















