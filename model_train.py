
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


############################################################################################
############ Feature Scaling

# create x,y
x = Df.drop(['target'], axis = True)
y = Df['target']

from sklearn.preprocessing import StandardScaler

# standardisation
scaler = StandardScaler()
# apply scaler
x = scaler.fit_transform(x)


############################################################################################
############ split data set

#startified
x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)

x_train.shape, y_train.shape
x_test.shape, y_test.shape


# check for stratify
y_train.value_counts()
y_test.value_counts() 


############################################################################################
############ knn training 

# models import
from sklearn.neighbors import KNeighborsClassifier

# classifier
classifier = KNeighborsClassifier()

# fit
classifier.fit(x_train, y_train,)

# predict
y_pred = classifier.predict(x_test)


# classifiacation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=wine.target_names)
disp.plot()


############################################################################################
############ hyperparameter tuninig

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# hyperparameters set
params = {'n_neighbors': np.arange(1,50),
          'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
          'metric':['minkowski', 'Euclidean']}


# Grid Search
grid = GridSearchCV(KNeighborsClassifier(),
                    param_grid=params,
                    cv=5,
                    scoring='accuracy')

grid.fit(x_train, y_train)


# show the best set
grid.best_estimator_
grid.best_params_
grid.best_score_


# predict
y_pred = grid.predict(x_test)


# classifiacation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



# Random Search
grid = RandomizedSearchCV(KNeighborsClassifier(),
                    params,
                    cv=5,
                    scoring='accuracy')

grid.fit(x_train, y_train)


# show the best set
grid.best_estimator_
grid.best_params_
grid.best_score_


# predict
y_pred = grid.predict(x_test)


# classifiacation report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

