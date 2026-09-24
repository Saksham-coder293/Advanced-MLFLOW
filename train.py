import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



# Load iris dataset
iris_data = load_iris()

# Convert to DataFrame
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['target'] = iris_data.target

# Now use it
print(iris.head())

X = iris.iloc[:,0:-1]
y = iris.iloc[:,-1]

#split the dataset into training and testing set 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#define the parameters for random forest model 

max_depth = 1
n_estimators = 100

#apply mlflow 
mlflow.set_experiment('iris-rf')
with mlflow.start.run():
    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(y_test)
    accuracy = accuracy_score(y_test,y_pred)


    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)


    #create a confusion matrix plot 
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize = (8,8))
    sns.heatmap(cm,annot = True , font = 'd',cmap = 'Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')


    #save the plot as an artifact
    plt.savefig('confusion_matrix.png')


    #mlflow code
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)
    mlflow.set_tag('another','saksham')
    mlflow.set_tag('model','random forest')

   #logging datasets
    train_df = X_train
    train_df['variety'] = y_train

    test_df = X_test
    test_df['variety'] = y_test

    train_df = mlflow.data.from_pandas(train_df)
    test_df = mlflow.data.from_pandas(test_df)

    mlflow.log_input(train_df,'train')
    mlflow.log_input(test_df,'validation')





print('accuracy',accuracy)