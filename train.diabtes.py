from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow



mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("file:./mlruns")

df  = pd.read_csv('health care diabetes.csv')

#splitting the dataset into features and targets 
X = df.drop('Outcome',axis = 1)
y = df['Outcome']

#splitting the dataset into training set and testing set 

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

#creating the random forest classifier 

rf = RandomForestClassifier(random_state = 42)

#defining the parameter grid for GridSearch CV
param_grid = {
    'n_estimators':[10,50,100],
    'max_depth' : [None,10,20,30]

}

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5,n_jobs = -1, verbose = 2)    #will show everything on console
mlflow.set_experiment('diabetes-rf-hp')

# fit the model first to compute the best params

with mlflow.start_run(run_name = "grid_search") as parent:

     grid_search.fit(X_train, y_train)

     #log all the children
     for i in range(0, len(grid_search.cv_results_['params'])):
          print(i)
          with mlflow.start_run(nested = True) as child:
               mlflow.log_params(grid_search.cv_results_['params'][i])
               mlflow.log_metric("accuracy",grid_search.cv_results_['mean_test_score'][i])

     best_params = grid_search.best_params_
     best_score = grid_search.best_score_

  # params
     mlflow.log_params(best_params)



    #metrics 
     mlflow.log_metric("accuracy",best_score)
    #data
     train_df = X_train
     train_df['Outcome'] = y_train

     mlflow.data.from_pandas(train_df)

     mlflow.log_input(train_df,"training")

     test_df = X_test
     test_df['Outcome'] =y_test

     test_df = X_test
     test_df['Outcome'] = y_test

     mlflow.data.from_pandas(test_df)
     mlflow.log_input(test_df,"validation")




  #source code 
     mlflow.log_artifact(__file__)



   #model
     mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")


   #tags 
     mlflow.set_tag("author","saksham")



  
#displaying the best parameters and best score 

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params)
print(best_score)

