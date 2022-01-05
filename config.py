hyp_param = {'max_depth':[None,6,8,10,12],
            'n_estimators': [50,100,200],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': [2],
            'min_samples_leaf':[0.01,0.03,0.05,0.1],
            'bootstrap': [True, False]}



# hyp_param = {'max_depth':[6],
#                  'n_estimators': [50],
#                  'max_features': ['sqrt'],
#                  'min_samples_split': [2],
#                  'min_samples_leaf':[1],
#                  'bootstrap': [False]}

pipeline_param ={'imputer_discrete__imputation_method': ['median'],
                 'imputer_continuous__imputation_method': ['median', 'mean'],
                 'imputer_categorical__imputation_method': ['frequent', 'missing'],
                 'outlier_detection__capping_method': ['gaussian', 'iqr'],
                 'outlier_detection__fold': [1.5, 3],
                 'Drop_constant_feature__tol': [1, 0.8],
                 'categorical_encoder__encoding_method': ['arbitrary'],
                 'categorical_encoder__drop_last': [True, False]
                 }