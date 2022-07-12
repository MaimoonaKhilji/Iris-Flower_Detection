import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')


df = pd.read_csv("iris_data.csv",encoding="latin-1")
df.head()
data = pd.DataFrame({'Max': df.head().max(), 'Min': df.head().min()})
from sklearn.model_selection import train_test_split
train =df.drop('Species',axis=1)
test = df['Species']
X_train, X_test, y_train, y_test = train_test_split(train,test,test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix

#print(confusion_matrix(y_test,pred))
#print(classification_report(y_test,pred))
error_rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print('WITH K=20')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
ax= plt.subplot()
sns.heatmap(confusion_matrix(y_test,pred), annot=True, ax = ax, fmt = 'g'); 
ax.set_title('Confusion Matrix', fontsize=20)

ax.xaxis.set_ticklabels(['setosa', 'versicolor','verginica'], fontsize = 12)
ax.xaxis.tick_top()

ax.yaxis.set_ticklabels(['setosa', 'versicolor','verginica'], fontsize = 12)
plt.show()


# ## Making a Predictive System

# In[115]:


input_data = (4.3,3.0,1.1,0.1)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print(input_data_reshaped)


prediction = knn.predict(input_data_reshaped)
print(prediction)


# In[116]:


import pickle


# ## saving the model

# In[118]:


filename = 'trained_model.sav'
pickle.dump(knn, open(filename, 'wb'))


# In[119]:


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[123]:



def iris_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    return prediction


# In[128]:


import streamlit as st
def main():
    st.title("Iris prediction")
    Sepal_length = st.text_input("Length of sepal")
    Sepal_width = st.text_input("Width of sepal")
    Petal_length = st.text_input("Length of Petal")
    Petal_width = st.text_input("Width of Petal")
    
    #code for prediction
    Species = ''
    
    #creating a button for prediction

    if st.button("Iris prediction"):
        Species=iris_prediction([Sepal_length,Sepal_width ,Petal_length,Petal_width])

    st.success(Species)

if __name__ == '__main__':
    main()






