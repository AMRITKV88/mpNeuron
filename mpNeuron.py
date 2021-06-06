# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Load Dataset

# %%
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target


# %%
print(X)
print(breast_cancer.feature_names)
print(X.shape)

# %% [markdown]
# Dataset preparation

# %%
# Dataset preparation
data_with_feature_name = pd.DataFrame(data = X, columns = breast_cancer.feature_names)
data = data_with_feature_name
data['class'] = Y

# %% [markdown]
# Train and Test Split

# %%
# Train and Test Split
from sklearn.model_selection import train_test_split
X_data = data.drop('class', axis=1)
Y_data = data['class']
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, stratify = Y_data, random_state = 1)

# %% [markdown]
# Binarization of data

# %%
# As MP Neuron takes binary data as input, so we have to binarize the input including train & test both
# Example of input data before binarization
from matplotlib import pyplot as plt
plt.plot(X_train.T, '*')
plt.xticks(rotation = 'vertical')
plt.show()


# %%
# The below shown code is just for an example of binarization on single column
# For actual training, the binarization need to be applied on all columns
# We have taken "mean area" and binarised on an apprx mean 1000 value of the column
binarised_on_the_besis_of_4th_col = X_train['mean area'].map(lambda x: 0 if x<1000 else 1)


# %%
# The plot shows that --> We have binaried the X_train on the basis of 4th col with a threshold value of 1000
plt.plot(binarised_on_the_besis_of_4th_col.T, '*')
plt.xticks(rotation = 'vertical')
plt.show()


# %%
# Now lets do it for entire X_train set using pd.cut --> which binarises the dataset automatically

# NB : You can play around with labels = [1,0] or [0,1] for both test and train then see the chnages 

binarised_X_train = X_train.apply(pd.cut, bins = 2, labels = [1,0])


# %%
plt.plot(binarised_X_train.T, '*')
plt.xticks(rotation = 'vertical')
plt.show()


# %%
# Now lets do it for entire X_test set using pd.cut --> which binarises the dataset automatically

# NB : You can play around with labels = [1,0] or [0,1] for both test and train then see the chnages 

binarised_X_test = X_test.apply(pd.cut, bins = 2, labels = [1,0])


# %%
plt.plot(binarised_X_test.T, '*')
plt.xticks(rotation = 'vertical')
plt.show()


# %%
# Take only the values into account and not the column names
binarised_X_train = binarised_X_train.values
binarised_X_test = binarised_X_test.values

# %% [markdown]
# Rnning MP Neron algorithm on our pre-processed dataset

# %%
# Check for which value of b, the MP Neuron gives highest accuracy
old_acucracy = 0
old_b = 0
for b in range(binarised_X_train.shape[1] + 1):
    accurate_pred = 0
    for i, j in zip(binarised_X_train, Y_train):
        y_pred = (np.sum(i) >= b)
        accurate_pred += (j == y_pred)
        accuracy = (accurate_pred/binarised_X_train.shape[0])*100
    print(b, accuracy)
    if old_acucracy < accuracy:
        old_acucracy = accuracy
        old_b = b
print("Highest training acuuracy for the value of b is :",old_b, "and highest training accuracy is :",old_acucracy)

# %% [markdown]
# MP Neuron Class

# %%
class mpNeuron:
    def __init__(self):
        self.b = None

    def model(self, x):
        return (np.sum(x) >=self.b)

    def predict(self, X):
        y = []
        for x in X:
            y_pred = self.model(x)
            y.append(y_pred)
        return y

    def fit(self, X, Y):
        accuracy = {}
        for b in range(binarised_X_train.shape[1] + 1):
            self.b = b
            accurate_pred = 0
            y_pred_train_list = self.predict(X)
            accuracy[b] = accuracy_score(y_pred_train_list, Y)
        best_b = max(accuracy, key = accuracy.get)
        print("Optimal accuracy found for b =",best_b,"and the accuracy value is =",accuracy[best_b])


# %%
mp_Neuron = mpNeuron()
mp_Neuron.fit(binarised_X_train, Y_train)


# %%



