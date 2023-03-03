import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Cross validation
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
#ensemble methods
from sklearn import linear_model, tree, ensemble, svm

#gradient coost classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def model(filename):
    df = pd.read_csv('C:\\Users\\jyoth\\Downloads\\TechGyan_Project\\customer_data.csv')
    #Conversion of data types
    df['TotalCharges'] = df['TotalCharges'].replace([' '],[0])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='raise')

    df['SeniorCitizen'] = df['SeniorCitizen'].replace([1,0],['Yes','No'])
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    #Categorical Variables
    df_cat = df.select_dtypes(include=['object'])
    df_cat = df_cat.drop(columns=['customerID','Churn'])

    cat_col = list(df_cat.columns.values)

    #Determining number of categories for the categorical values

    object_cols = [col for col in df.columns if df[col].dtype == "object"]

    # Get number of unique entries in each column with categorical data

    object_nunique = list(map(lambda col: df[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    #We apply one hot encoding for feature engineering 
    for var in cat_col:
        cat_list = 'var' + '_' + 'var'
        cat_list = pd.get_dummies(df[var], prefix=var)
        df_New = pd.concat([df,cat_list],axis = 1)
        df = df_New

    data_vars = df.columns.values.tolist()

    to_keep = [i for i in data_vars if i not in cat_col]

    df_final = df[to_keep]


    df_final['Churn'] = df_final['Churn'].replace(['Yes','No'],[1,0])

    df_model =df_final.drop(columns = ['customerID'])

    label = np.array(df_model['Churn'])

    feature_df = df_model.drop('Churn', axis=1)

    #A list of features
    feature_list = list(feature_df.columns)

    # change the feature dataframe to an array
    feature= np.array(feature_df)

    feature_train, feature_test, label_train, label_test = train_test_split(feature, label,test_size = 0.2, random_state= 42)

    #Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)

    # Train the model using the training data
    rf.fit(feature_train, label_train)
    y_pred = rf.predict(feature_test)


    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



    gbc = ensemble.GradientBoostingClassifier(random_state=42) 
    gbc.fit(feature, label)

    col_sorted_by_importance=gbc.feature_importances_.argsort()
    feat_imp=pd.DataFrame({
        'cols':feature_df.columns[col_sorted_by_importance],
        'imps':gbc.feature_importances_[col_sorted_by_importance]
    })

    model = GradientBoostingClassifier(random_state=1)
    space = dict()
    space['n_estimators'] = [10, 100, 500]
    space['max_features'] = [2, 4, 6]
    search = GridSearchCV(model,space, scoring='accuracy',cv = kf, refit=True)
    result = search.fit(feature_train, label_train)

    best_model = result.best_estimator_


    df = pd.read_csv('C:\\Users\\jyoth\\Downloads\\TechGyan_Project\\test.csv')
    f = df
    #Conversion of data types
    df['TotalCharges'] = df['TotalCharges'].replace([' '],[0])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='raise')

    df['SeniorCitizen'] = df['SeniorCitizen'].replace([1,0],['Yes','No'])
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    #Categorical Variables
    df_cat = df.select_dtypes(include=['object'])
    df_cat = df_cat.drop(columns=['customerID'])

    cat_col = list(df_cat.columns.values)

    #Determining number of categories for the categorical values

    object_cols = [col for col in df.columns if df[col].dtype == "object"]

    # Get number of unique entries in each column with categorical data

    object_nunique = list(map(lambda col: df[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    #We apply one hot encoding for feature engineering 
    for var in cat_col:
        cat_list = 'var' + '_' + 'var'
        cat_list = pd.get_dummies(df[var], prefix=var)
        df_New = pd.concat([df,cat_list],axis = 1)
        df = df_New

    data_vars = df.columns.values.tolist()

    to_keep = [i for i in data_vars if i not in cat_col]

    df_final = df[to_keep]


    df_model =df_final.drop(columns = ['customerID'])

    feature_df = df_model
    #A list of features
    feature_list = list(feature_df.columns)

    # change the feature dataframe to an array
    feature= np.array(feature_df)
    

    yhat = best_model.predict(feature)

    rescmp=pd.DataFrame(f)
    rescmp['Predicted Churn']=yhat
    rescmp['Predicted Churn'] = rescmp['Predicted Churn'].replace([1,0],['Yes','No'])
    rescmp.to_csv('C:\\Users\\jyoth\\Downloads\\TechGyan_Project\\result.csv')

    return 1
