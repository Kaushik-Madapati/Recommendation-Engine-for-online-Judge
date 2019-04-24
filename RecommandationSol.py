# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 11:11:35 2019

@author: nmadapati
"""

import pandas as pd 
import numpy as np

import keras


from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import  classification_report
from sklearn.metrics import  confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, Adagrad, RMSprop



def level_num(level) :
     value = level
     if value == 'A':
         return 1
     elif value == 'B':
         return 2
     elif value == 'C':
         return 3
     elif value == 'D':
         return 4
     elif value == 'E':
         return 5
     elif value == 'F':
         return 6
     elif value == 'G':
         return 7
     elif value == 'H':
         return 8
     elif value == 'I':
         return 9
     elif value == 'J':
         return 10
     elif value == 'K':
         return 11
     elif value == 'L':
         return 12
     elif value == 'M':
         return 13
     elif value == 'N':
         return 14
     else :
        return 15

def rank_num(level) :
     value = level
     if value == 'intermediate':
         return 2
     elif value == 'beginner':
         return 1
     elif value == 'expert':
         return 4
     else :
        return 3
    
    
def cleaningData(data) :
    
    ####################################################################
    #Cleaning data 
    #
    ##########################################################################
    
 
    data['rank'] = data['rank'].apply(rank_num)
    data['level_type'] = data['level_type'].apply(level_num)
    drop_items = [ 'problem_id', 'country',   'user_id', 'problem_solved',
              'submission_count', 'rating', 'max_rating', 'registration_time_seconds', 
              'last_online_time_seconds' , 'tags', 'points']
    
    #    
    data.drop(drop_items, axis=1, inplace= True)

    
    return data




#############################################################################
def PCAModel(data ) :
    pca = PCA(n_components = 2)
    id_data = data['user_id']
    data.drop(['user_id'], axis=1, inplace= True)
    pca.fit(data)
    pc12 = pca.transform(data)
    data = pd.DataFrame(pc12, columns = ['PC1', 'PC2'])
    data = pd.concat([data, id_data], axis =1)
    return data
########################################################################



# Comparing different Classifieers
####################################################
def ClassifierComparisionModel(data=0):
    
  
    
    names = [ "Nearest Neighbors", "Decision Tree", 
         "Naive Bayes", "QDA", 'XGBoost']

    classifiers = [
            KNeighborsClassifier(3),
            DecisionTreeClassifier(max_depth=5),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            XGBClassifier()]
    anova_filter = SelectKBest(f_regression, k='all')
    
    for name, clf in zip(names, classifiers):
      
        estimator = make_pipeline(anova_filter, clf)
        estimator.fit(X_train, Y_train)
        print("Model name :", clf)
        predit = estimator.predict(X_test)
        print(confusion_matrix(Y_test, predit))
        print(classification_report(Y_test, predit))

#######################################################################
### Final model 
#######################################################################
def FinalModel(data = 0) :
           
    anova_filter = SelectKBest(f_regression, k='all')
    knn =   KNeighborsClassifier()
    estimator = make_pipeline(anova_filter, knn)
    estimator.fit(X_train, Y_train)
    return estimator
      

#########################################################################      
def NeuralNetmodel(lr=0):
    
   
   
    
    params = {
            'input_layer_width': 200,  #50
            'hidden_layer_count': 5,  #20
            'hidden_layer_width':100  #10
            # 'learning_rate': hp.normal('learning_rate', 0.001, 0.00075)
    }
    
    model = Sequential()
    model.add(Dense(params['input_layer_width'],
                     activation='relu', 
                     input_dim=13))
    for l in range(params['hidden_layer_count']):
        model.add(Dense(params['hidden_layer_width'],
                        activation='relu'))

    model.add(Dense(7, 
                    activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adagrad(lr=0.001),
                  metrics=['accuracy'])
    
    h = model.fit(X_train, Y_train, batch_size=32,epochs=50, validation_split=0.3 , verbose=1)
    loss_value = pd.DataFrame(h.history['loss'])
    
    
    loss_value.plot()
    
    return model
        
def TestModelWithTrack(model) :
    
    predit = model.predict(X_test)
    
    print(classification_report(Y_test.argmax(axis=1), predit.argmax(axis=1)))

    
#######################################################################
#### Read data
######################################################################
    
###################################
# Global data 
(X_train, X_test), (Y_train, Y_test) = (None, None), (None, None)

if __name__ == "__main__":
    

    problem = pd.read_csv("problem_data.csv")
    
    submission_data_raw = pd.read_csv("train_submissions.csv")
    
    submission_data = submission_data_raw.drop('attempts_range', axis=1)
    
    
    user_data = pd.read_csv("user_data.csv")
    user_data['success_ratio'] = user_data['problem_solved']/user_data['submission_count']
    user_data['rank_ratio'] = user_data['rating']/user_data['max_rating']
    
    df_raw = pd.merge(submission_data, user_data, on = 'user_id')
    df_raw = pd.merge(problem, df_raw, on = 'problem_id')
    
    print("X_train" , df_raw.info())
    df_raw = cleaningData(df_raw)
    
    
    y = submission_data_raw['attempts_range']
    ##
    
    
    scaler = StandardScaler()
    scaler.fit(df_raw)
    scaler_input = scaler.transform(df_raw)
    x= scaler_input
    
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.33, random_state=43)
    
    model= FinalModel()
    
    predit = model.predict(X_test)
    print(predit)
    print(confusion_matrix(Y_test, predit))
    print(classification_report(Y_test, predit)) 
    
    
    test_data_raw = pd.read_csv("test_submissions.csv")
    test_data = test_data_raw.drop(['ID'], axis=1)
    
    
    test_df = pd.merge(test_data, user_data, on = 'user_id')
    test_problem_submission = pd.merge(problem, test_df, on = 'problem_id')
    
    ##
    ###
    print("X_test", test_problem_submission.info())
    test_clean_Data = cleaningData(test_problem_submission)
    
    test_predit = model.predict(test_clean_Data)
    submission = pd.DataFrame({ "ID": test_data_raw["ID"],      
                               "attempts_range": test_predit  })
    #    
    #
    submission.to_csv('Recommendation_sol.csv', index=False)

