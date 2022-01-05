import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for the model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from feature_engine import imputation
from feature_engine import encoding
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from feature_engine.outliers import Winsorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from feature_engine.discretisation import EqualFrequencyDiscretiser
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from feature_engine.selection import DropConstantFeatures
from feature_engine.encoding import OrdinalEncoder
import math


df = pd.read_csv(r'C:\Users\AH41803\OneDrive - Anthem\Documents\preprocess_sklearn_pipeline\data2.csv')
y=df['DEATH_EVENT']
X=pd.DataFrame(df.drop(['DEATH_EVENT'],axis=1))

hyp_param = {'max_depth':[None,6,8,10,12],
             'n_estimators': [100,200],
             'max_features': ['sqrt', 'log2', None],
             'min_samples_split': [2],
             'min_samples_leaf':[1,50,0.01,0.02,0.05],
             'bootstrap': [True, False]}
scoring_method = 'accuracy'

col=len(X.columns)
if col<10:
    feature_sel_list = [math.floor(col/2), col]
else:
    feature_sel_list = [math.floor(col / 3),2*(math.floor(col / 3)), col]


categorical = [var for var in X.columns if df[var].dtype == 'O']
discrete = [var for var in X.columns if df[var].dtype != 'O' and (df[var].nunique())<=10]
continuous = [var for var in X.columns if df[var].dtype != 'O' and var not in discrete]

X_train, X_test, y_train, y_test = train_test_split(
    X,  # predictors
    y, # target
    test_size=0.1,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility




#class_wight calculation
from sklearn.utils import class_weight
classes = np.unique(y_train)
cw = class_weight.compute_class_weight('balanced', classes, y_train)
weights = dict(zip(classes,cw))

if len(categorical)!=0 and len(discrete)!=0 and len(continuous)!=0:
    pipe = Pipeline([
        ('imputer_discrete',
         imputation.MeanMedianImputer(variables=discrete)),
        ('imputer_categorical',
         imputation.CategoricalImputer(variables=categorical)),
        ('imputer_continuous',
         imputation.MeanMedianImputer(variables=continuous)),
        ('outlier_detection',
         Winsorizer(variables=continuous)),
        ('categorical_encoder',
         encoding.OneHotEncoder(variables=categorical)),
        ('Drop_constant_feature',
         DropConstantFeatures(missing_values='ignore')),
        ('feature_transform',
         SklearnTransformerWrapper(transformer=StandardScaler())),
        ('feature_selection',
         RFE(estimator=DecisionTreeClassifier())),

        # random forest
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = [{
        # try different feature engineering parameters
        'imputer_discrete__imputation_method': ['median'],
        'imputer_continuous__imputation_method': ['median', 'mean'],
        'imputer_categorical__imputation_method': ['frequent', 'missing'],
        'outlier_detection__capping_method': ['gaussian', 'iqr'],
        'outlier_detection__fold': [1.5, 2, 3],
        'categorical_encoder__drop_last': [True, False],
        'Drop_constant_feature__tol': [1, 0.8],
        'feature_transform__variables': [[], continuous],
        'feature_selection__n_features_to_select': feature_sel_list,


        # try different random forest tree model paramenters
        'rf__max_depth': hyp_param['max_depth'],
        'rf__n_estimators': hyp_param['n_estimators'],
        'rf__class_weight': [weights, None],
        'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
        'rf__min_samples_split': hyp_param['min_samples_split'],
        'rf__max_features': hyp_param['max_features'],
        'rf__bootstrap': hyp_param['bootstrap']
    },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder__drop_last': [True, False],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (SklearnTransformerWrapper(transformer=MinMaxScaler()),),
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (SklearnTransformerWrapper(transformer=MinMaxScaler()),),
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (EqualFrequencyDiscretiser(),),
            'feature_transform__variables': [continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder__drop_last': [True, False],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (EqualFrequencyDiscretiser(),),
            'feature_transform__variables': [continuous],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        }]
elif len(categorical)==0 and len(discrete)!=0 and len(continuous)!=0:
    pipe = Pipeline([
        ('imputer_discrete',
         imputation.MeanMedianImputer(variables=discrete)),
        ('imputer_continuous',
         imputation.MeanMedianImputer(variables=continuous)),
        ('outlier_detection',
         Winsorizer(variables=continuous)),
        ('Drop_constant_feature',
         DropConstantFeatures(missing_values='ignore')),
        ('feature_transform',
         SklearnTransformerWrapper(transformer=StandardScaler())),
        ('feature_selection',
         RFE(estimator=DecisionTreeClassifier())),

        # random forest
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = [{
        # try different feature engineering parameters
        'imputer_discrete__imputation_method': ['median'],
        'imputer_continuous__imputation_method': ['median', 'mean'],
        'outlier_detection__capping_method': ['gaussian', 'iqr'],
        'outlier_detection__fold': [1.5, 2, 3],
        'Drop_constant_feature__tol': [1, 0.8],
        'feature_transform__variables': [[], continuous],
        'feature_selection__n_features_to_select': feature_sel_list,

        # try different random forest tree model paramenters
        'rf__max_depth': hyp_param['max_depth'],
        'rf__n_estimators': hyp_param['n_estimators'],
        'rf__class_weight': [weights, None],
        'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
        'rf__min_samples_split': hyp_param['min_samples_split'],
        'rf__max_features': hyp_param['max_features'],
        'rf__bootstrap': hyp_param['bootstrap']
    },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (SklearnTransformerWrapper(transformer=MinMaxScaler()),),
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (SklearnTransformerWrapper(transformer=MinMaxScaler()),),
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (EqualFrequencyDiscretiser(),),
            'feature_transform__variables': [continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (EqualFrequencyDiscretiser(),),
            'feature_transform__variables': [continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        }]
elif len(categorical)==0 and len(discrete)==0 and len(continuous)!=0:

    pipe = Pipeline([
        ('imputer_continuous',
         imputation.MeanMedianImputer(variables=continuous)),
        ('outlier_detection',
         Winsorizer(variables=continuous)),
        ('Drop_constant_feature',
         DropConstantFeatures(missing_values='ignore')),
        ('feature_transform',
         SklearnTransformerWrapper(transformer=StandardScaler())),
        ('feature_selection',
         RFE(estimator=DecisionTreeClassifier())),

        # random forest
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = [{
        # try different feature engineering parameters
        'imputer_continuous__imputation_method': ['median', 'mean'],
        'outlier_detection__capping_method': ['gaussian', 'iqr'],
        'outlier_detection__fold': [1.5, 2, 3],
        'Drop_constant_feature__tol': [1, 0.8],
        'feature_transform__variables': [[], continuous],
        'feature_selection__n_features_to_select': feature_sel_list,

        # try different random forest tree model paramenters
        'rf__max_depth': hyp_param['max_depth'],
        'rf__n_estimators': hyp_param['n_estimators'],
        'rf__class_weight': [weights, None],
        'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
        'rf__min_samples_split': hyp_param['min_samples_split'],
        'rf__max_features': hyp_param['max_features'],
        'rf__bootstrap': hyp_param['bootstrap']
    },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (SklearnTransformerWrapper(transformer=MinMaxScaler()),),
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (SklearnTransformerWrapper(transformer=MinMaxScaler()),),
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (EqualFrequencyDiscretiser(),),
            'feature_transform__variables': [continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (EqualFrequencyDiscretiser(),),
            'feature_transform__variables': [continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        }]
elif len(categorical)==0 and len(discrete)!=0 and len(continuous)==0:

    pipe = Pipeline([
        ('imputer_discrete',
         imputation.MeanMedianImputer(variables=discrete)),
        ('Drop_constant_feature',
         DropConstantFeatures(missing_values='ignore')),
        ('feature_selection',
         RFE(estimator=DecisionTreeClassifier())),

        # random forest
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = [{
        # try different feature engineering parameters
        'imputer_discrete__imputation_method': ['median'],
        'Drop_constant_feature__tol': [1, 0.8],
        'feature_selection__n_features_to_select': feature_sel_list,

        # try different random forest tree model paramenters
        'rf__max_depth': hyp_param['max_depth'],
        'rf__n_estimators': hyp_param['n_estimators'],
        'rf__class_weight': [weights, None],
        'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
        'rf__min_samples_split': hyp_param['min_samples_split'],
        'rf__max_features': hyp_param['max_features'],
        'rf__bootstrap': hyp_param['bootstrap']
    },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        }]
elif len(categorical)!=0 and len(discrete)!=0 and len(continuous)==0:
    pipe = Pipeline([
        ('imputer_discrete',
         imputation.MeanMedianImputer(variables=discrete)),
        ('imputer_categorical',
         imputation.CategoricalImputer(variables=categorical)),
        ('categorical_encoder',
         encoding.OneHotEncoder(variables=categorical)),
        ('Drop_constant_feature',
         DropConstantFeatures(missing_values='ignore')),
        ('feature_selection',
         RFE(estimator=DecisionTreeClassifier())),

        # random forest
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = [{
        # try different feature engineering parameters
        'imputer_discrete__imputation_method': ['median'],
        'imputer_categorical__imputation_method': ['frequent', 'missing'],
        'categorical_encoder__drop_last': [True, False],
        'Drop_constant_feature__tol': [1, 0.8],
        'feature_selection__n_features_to_select': feature_sel_list,


        # try different random forest tree model paramenters
        'rf__max_depth': hyp_param['max_depth'],
        'rf__n_estimators': hyp_param['n_estimators'],
        'rf__class_weight': [weights, None],
        'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
        'rf__min_samples_split': hyp_param['min_samples_split'],
        'rf__max_features': hyp_param['max_features'],
        'rf__bootstrap': hyp_param['bootstrap']
    },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder__drop_last': [True, False],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_discrete__imputation_method': ['median'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder__drop_last': [True, False],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        }]
elif len(categorical)!=0 and len(discrete)==0 and len(continuous)==0:
    pipe = Pipeline([
        ('imputer_categorical',
         imputation.CategoricalImputer(variables=categorical)),
        ('categorical_encoder',
         encoding.OneHotEncoder(variables=categorical)),
        ('Drop_constant_feature',
         DropConstantFeatures(missing_values='ignore')),
        ('feature_selection',
         RFE(estimator=DecisionTreeClassifier())),

        # random forest
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = [{
        # try different feature engineering parameters
        'imputer_categorical__imputation_method': ['frequent', 'missing'],
        'categorical_encoder__drop_last': [True, False],
        'Drop_constant_feature__tol': [1, 0.8],
        'feature_selection__n_features_to_select': feature_sel_list,


        # try different random forest tree model paramenters
        'rf__max_depth': hyp_param['max_depth'],
        'rf__n_estimators': hyp_param['n_estimators'],
        'rf__class_weight': [weights, None],
        'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
        'rf__min_samples_split': hyp_param['min_samples_split'],
        'rf__max_features': hyp_param['max_features'],
        'rf__bootstrap': hyp_param['bootstrap']
    },
        {
            # try different feature engineering parameters
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder__drop_last': [True, False],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'categorical_encoder__drop_last': [True, False],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        }]
elif len(categorical)!=0 and len(discrete)==0 and len(continuous)!=0:

    pipe = Pipeline([
        ('imputer_categorical',
         imputation.CategoricalImputer(variables=categorical)),
        ('imputer_continuous',
         imputation.MeanMedianImputer(variables=continuous)),
        ('outlier_detection',
         Winsorizer(variables=continuous)),
        ('categorical_encoder',
         encoding.OneHotEncoder(variables=categorical)),
        ('Drop_constant_feature',
         DropConstantFeatures(missing_values='ignore')),
        ('feature_transform',
         SklearnTransformerWrapper(transformer=StandardScaler())),
        ('feature_selection',
         RFE(estimator=DecisionTreeClassifier())),

        # random forest
        ('rf', RandomForestClassifier(random_state=0))
    ])
    param_grid = [{
        # try different feature engineering parameters
        'imputer_continuous__imputation_method': ['median', 'mean'],
        'imputer_categorical__imputation_method': ['frequent', 'missing'],
        'outlier_detection__capping_method': ['gaussian', 'iqr'],
        'outlier_detection__fold': [1.5, 2, 3],
        'categorical_encoder__drop_last': [True, False],
        'Drop_constant_feature__tol': [1, 0.8],
        'feature_transform__variables': [[], continuous],
        'feature_selection__n_features_to_select': feature_sel_list,


        # try different random forest tree model paramenters
        'rf__max_depth': hyp_param['max_depth'],
        'rf__n_estimators': hyp_param['n_estimators'],
        'rf__class_weight': [weights, None],
        'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
        'rf__min_samples_split': hyp_param['min_samples_split'],
        'rf__max_features': hyp_param['max_features'],
        'rf__bootstrap': hyp_param['bootstrap']
    },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder__drop_last': [True, False],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (SklearnTransformerWrapper(transformer=MinMaxScaler()),),
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (SklearnTransformerWrapper(transformer=MinMaxScaler()),),
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform__variables': [[], continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder': (encoding.OrdinalEncoder(variables=categorical),),
            'categorical_encoder__encoding_method': ['arbitrary'],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (EqualFrequencyDiscretiser(),),
            'feature_transform__variables': [continuous],
            'feature_selection__n_features_to_select': feature_sel_list,

            # try different random forest tree model parameters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        },
        {
            # try different feature engineering parameters
            'imputer_continuous__imputation_method': ['median', 'mean'],
            'imputer_categorical__imputation_method': ['frequent', 'missing'],
            'outlier_detection__capping_method': ['gaussian', 'iqr'],
            'outlier_detection__fold': [1.5, 2, 3],
            'categorical_encoder__drop_last': [True, False],
            'Drop_constant_feature__tol': [1, 0.8],
            'feature_transform': (EqualFrequencyDiscretiser(),),
            'feature_transform__variables': [continuous],
            'feature_selection__n_features_to_select': feature_sel_list,


            # try different random forest tree model paramenters
            'rf__max_depth': hyp_param['max_depth'],
            'rf__n_estimators': hyp_param['n_estimators'],
            'rf__class_weight': [weights, None],
            'rf__min_samples_leaf': hyp_param['min_samples_leaf'],
            'rf__min_samples_split': hyp_param['min_samples_split'],
            'rf__max_features': hyp_param['max_features'],
            'rf__bootstrap': hyp_param['bootstrap']
        }]


rskf = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 1, random_state = 42)
grid_search = GridSearchCV(pipe, param_grid,
                           cv=rskf, n_jobs=-1, scoring=scoring_method)
grid_search.fit(X_train, y_train)

grid_predictions = grid_search.predict(X_test)

# print classification report
print(classification_report(y_test, grid_predictions))

results2=pd.DataFrame(grid_search.cv_results_)
results2.to_excel('result3.xlsx')
print(grid_search.best_params_)


