#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from uuid import uuid4
import itertools
import copy


# In[ ]:


df = pd.read_csv(r'C:\Users\AH41803\OneDrive - Anthem\Documents\preprocess_sklearn_pipeline\horse-colic.csv',na_values='?',
                 sep=' ', index_col=False,names=list(range(28)))


# In[72]:


class preprocessing:
    def __init__(self,df,path):
        self.df=df
        self.path=path
        self.config=[]
    def column_unique(self):
        print(self.df_new.nunique())
    def missing_value(self):
        for i in (self.df_new.columns):
            n_miss = self.df_new[i].isnull().sum()
            perc = n_miss / self.df_new.shape[0] * 100
            print(i," Missing Value :",n_miss," percentage",perc)
    def withput_missing_value_in_target(self):
        self.df_new=self.df[self.df[self.df.columns[-1]].notna()]
    def numerical_var(self):
        self.numeric_var = [key for key in dict(self.df_new.dtypes) 
                       if dict(self.df_new.dtypes)[key]
                       in ['float64', 'float32', 'int32', 'int64']]
        self.numeric_df=self.df_new[self.numeric_var]        
    def categorical_var(self):
        self.cat_var = [key for key in dict(self.df_new.dtypes)
                   if dict(self.df_new.dtypes)[key] in ['object']]
        self.categorical_df=self.df_new[self.cat_var]
    def continuous_var(self):
        self.cont_var=[a for a in self.numeric_var 
                       if (self.df_new[a].nunique()/self.df_new.shape[0])>0.05]
    def discrete_var(self):
        self.disc_var=[a for a in self.numeric_var 
                       if (self.df_new[a].nunique()/self.df_new.shape[0])<=0.05]
    def knn(self):
        from sklearn.impute import KNNImputer
        from sklearn.impute import SimpleImputer
        self.configuration_knn=[]           
        weights_list = ['uniform', 'distance']
        neighbors_list = [2,4,6]
        knn_list = list(itertools.product(weights_list,neighbors_list))
        for ele in knn_list:
            preprocess_parameters = {}
            preprocess_parameters['knn_weights'] = ele[0]
            preprocess_parameters['knn_neighbors'] = ele[1]
            preprocess_parameters['imputation'] = "KNNImputer(cont.),Simple(discrete)"
            preprocess_parameters['simple_strategy_numerical_imputation']='median'
            preprocess_parameters['simple_strategy_categorical_imputation']='most_frequent'
            try:
                imputer = KNNImputer(n_neighbors=ele[1], weights=ele[0])
                knnimp = imputer.fit_transform(self.df_new[self.cont_var])
                df_knnimp = pd.DataFrame(knnimp)
                df_knnimp.set_axis(self.cont_var, axis=1, inplace=True)
            except:
                continue                
            try:
                imputer2 = SimpleImputer(strategy='median')
                disc_imp = imputer2.fit_transform(self.df_new[self.disc_var])
                df_simp=pd.DataFrame(disc_imp)
                df_simp.set_axis(self.disc_var,axis=1,inplace=True)
            except:
                continue
            try:                
                imputer3 = SimpleImputer(strategy='most_frequent')
                cat_imp = imputer3.fit_transform(self.categorical_df.values)
                df_simp2=pd.DataFrame(data = cat_imp, columns = self.cat_var)
            except:
                continue

            knnimp_df = pd.concat([df_knnimp,df_simp,df_simp2], axis=1)
            path = str(self.path) + "data_set_" +str(uuid4()) + str(".csv")
            knnimp_df.to_csv(path,index=False)
            config1 = {}
            config1["dataset_path"]= path
            config1["is_best"] = False
            config1 = {"dataset_path": path, "preprocess_parameters":preprocess_parameters,
                         "is_best":False}
            self.configuration_knn.append(config1)           
        self.config.extend(self.configuration_knn)
    def withput_imputation_column(self):
        df_without_imputation = self.df_new.dropna(axis=1)
        path_without_imputation = str(self.path) + "data_set_"+str(uuid4()) + str(".csv")
        df_without_imputation.to_csv(path_without_imputation, index = False)
        self.config_wic  = {}
        self.config_wic["dataset_path"] = path_without_imputation
        self.config_wic["preprocess_parameters"]= {"imputation": None}
        self.config_wic["is_best"]= False
        self.config.append(self.config_wic)
    def withput_imputation_row(self):
        df_without_imputation = self.df_new.dropna(axis=0)
        path_without_imputation = str(self.path) + "data_set_"+str(uuid4()) + str(".csv")
        df_without_imputation.to_csv(path_without_imputation, index = False)
        self.config_wir  = {}
        self.config_wir["dataset_path"] = path_without_imputation
        self.config_wir["preprocess_parameters"]= {"imputation": None}
        self.config_wir["is_best"]= False
        self.config.append(self.config_wir)
    def standardscaler(self):
        self.config_transform=[]
        self.config_before_transform = copy.deepcopy(self.config)
        for l in range(len(self.config_before_transform)):
            file=list(self.config_before_transform[l].values())[0]
            df_input = pd.read_csv(file)
            numeric_var = [key for key in dict(df_input.dtypes) 
                           if dict(df_input.dtypes)[key]
                           in ['float64', 'float32', 'int32', 'int64']]
            cat_var = [key for key in dict(df_input.dtypes)
                       if dict(df_input.dtypes)[key] in ['object']]
            cont_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])>0.05]
            disc_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])<=0.05]
            
            if len(cont_var)!=0:
                ##StandardScaler
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_num = scaler.fit_transform(df_input[cont_var])
                scaled_df = pd.DataFrame(scaled_num)
                scaled_df.set_axis(cont_var, axis=1, inplace=True)
                standarisation_df = pd.concat([scaled_df, df_input[disc_var],df_input[cat_var]], axis=1)
                path = str(self.path) + "data_set_"+str(uuid4()) + str(".csv")
                standarisation_df.to_csv(path,index=False)
                #import pdb;pdb.set_trace()
                config = {}
                config["dataset_path"]= path
                config["preprocess_parameters"] = copy.deepcopy(self.config_before_transform[l]["preprocess_parameters"])
                config["preprocess_parameters"]["scalar"] = "StandardScaler"
                config["is_best"] = False
                self.config_transform.append(config)
        self.config.extend(self.config_transform)
    def normalizer(self):
        self.config_normalize=[]
        for l in range(len(self.config_before_transform)):
            file=list(self.config_before_transform[l].values())[0]
            df_input = pd.read_csv(file)
            numeric_var = [key for key in dict(df_input.dtypes) 
                           if dict(df_input.dtypes)[key]
                           in ['float64', 'float32', 'int32', 'int64']]
            cat_var = [key for key in dict(df_input.dtypes)
                       if dict(df_input.dtypes)[key] in ['object']]
            cont_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])>0.05]
            disc_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])<=0.05]
            
            if len(cont_var)!=0:
                ##Normalizer
                norm_list = ['l1', 'l2', 'max']
                for n in norm_list:
                    from sklearn.preprocessing import Normalizer
                    normalise = Normalizer(norm = n)
                    X_normalised = normalise.fit_transform(df_input[cont_var])
                    df_normal = pd.DataFrame(X_normalised)
                    df_normal.set_axis(cont_var, axis=1, inplace=True)
                    normalised_df = pd.concat([df_normal, df_input[disc_var],df_input[cat_var]], axis=1)
                    path = str(self.path) + "data_set_"+str(uuid4()) + str(".csv")
                    normalised_df.to_csv(path,index=False)
                    config = {}
                    config["dataset_path"]= path
                    config["preprocess_parameters"] = copy.deepcopy(self.config_before_transform[l]["preprocess_parameters"])
                    config["preprocess_parameters"]["scalar"] = "Normalization"
                    config["preprocess_parameters"]["norm"] = n
                    config["is_best"] = False
                    self.config_normalize.append(config)
        self.config.extend(self.config_normalize)
        
    def minmaxscaler(self):
        self.config_minmax=[]
        for l in range(len(self.config_before_transform)):
            file=list(self.config_before_transform[l].values())[0]
            df_input = pd.read_csv(file)
            numeric_var = [key for key in dict(df_input.dtypes) 
                           if dict(df_input.dtypes)[key]
                           in ['float64', 'float32', 'int32', 'int64']]
            cat_var = [key for key in dict(df_input.dtypes)
                       if dict(df_input.dtypes)[key] in ['object']]
            cont_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])>0.05]
            disc_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])<=0.05]
            
            if len(cont_var)!=0:
            #minmaxscaler
                from sklearn.preprocessing import MinMaxScaler
                min_max_scaler = MinMaxScaler()
                minmax = min_max_scaler.fit_transform(df_input[cont_var])
                df_minmax = pd.DataFrame(minmax)
                df_minmax.set_axis(cont_var, axis=1, inplace=True)
                minmaxscaler_df = pd.concat([df_minmax, df_input[disc_var],df_input[cat_var]], axis=1)
                path = str(self.path) + "data_set_"+str(uuid4()) + str(".csv")
                minmaxscaler_df.to_csv(path,index=False)
                config = {}
                config["dataset_path"]= path
                config["preprocess_parameters"] = copy.deepcopy(self.config_before_transform[l]["preprocess_parameters"])
                config["preprocess_parameters"]["scalar"] = "MixMaxScalar"
                config["is_best"] = False
                self.config_minmax.append(config)
            
        self.config.extend(self.config_minmax)
        
    def ordinal_encoding(self):
        self.config_encoding=[]
        self.config_before_encoding = copy.deepcopy(self.config)
        for l in range(len(self.config_before_encoding)):
            file=list(self.config_before_encoding[l].values())[0]
            df_input = pd.read_csv(file)
            numeric_var = [key for key in dict(df_input.dtypes) 
                           if dict(df_input.dtypes)[key]
                           in ['float64', 'float32', 'int32', 'int64']]
            cat_var = [key for key in dict(df_input.dtypes)
                       if dict(df_input.dtypes)[key] in ['object']]
            cont_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])>0.05]
            disc_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])<=0.05]
            
            #ordinal Encoding
            from sklearn.preprocessing import OrdinalEncoder
            ord = OrdinalEncoder()
            ordenc = ord.fit_transform(df_input[cat_var])
            ordenc_df = pd.DataFrame(ordenc)
            label_names = []
            for j in cat_var:
                ordenc_df_columns = str(j) + str('_label')
                label_names.append(ordenc_df_columns)
            ordenc_df.set_axis(label_names, axis=1, inplace=True)
            ordenc2 = ord.fit_transform(df_input[disc_var])
            ordenc_df2 = pd.DataFrame(ordenc2)
            label_names2 = []
            for j in disc_var:
                ordenc_df_columns2 = str(j) + str('_label')
                label_names2.append(ordenc_df_columns2)
            ordenc_df2.set_axis(label_names2, axis=1, inplace=True)
            ordinal_df = pd.concat([df_input[cont_var], ordenc_df2, ordenc_df], axis=1)
            path = str(self.path) + "data_set_"+str(uuid4()) + str(".csv")
            ordinal_df.to_csv(path,index=False)
            config = {}
            config["dataset_path"]= path
            config["preprocess_parameters"] = copy.deepcopy(self.config_before_encoding[l]["preprocess_parameters"])
            config["preprocess_parameters"]["encoding"] = "ordinal_encoding"
            config["is_best"] = False
            self.config_encoding.append(config)
        self.config.extend(self.config_encoding)
        
    
    def dummy_encoding(self):
        self.config_dummy=[]
        for l in range(len(self.config_before_encoding)):
            file=list(self.config_before_encoding[l].values())[0]
            df_input = pd.read_csv(file)
            numeric_var = [key for key in dict(df_input.dtypes) 
                           if dict(df_input.dtypes)[key]
                           in ['float64', 'float32', 'int32', 'int64']]
            cat_var = [key for key in dict(df_input.dtypes)
                       if dict(df_input.dtypes)[key] in ['object']]
            cont_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])>0.05]
            disc_var=[a for a in numeric_var 
                      if (df_input[a].nunique()/df_input.shape[0])<=0.05]
            
            #dummy encoding
            dummy = []
            for k in cat_var:
                dum_df = pd.get_dummies(df_input[cat_var], columns=[k], prefix=["Type_is"])
                dummy.append(dum_df)
            try: 
                dummy_concat = pd.concat(dummy, axis=1)
                dummy_concat = dummy_concat.loc[:,~dummy_concat.columns.duplicated()]
                dummy_concat.drop(columns = cat_var, axis =1, inplace=True)
            except:
                continue
            
            dummy2=[]
            for k in disc_var:
                dum_df2 = pd.get_dummies(df_input[disc_var], columns=[k], prefix=["Type_is"])
                dummy2.append(dum_df2)
            try:
                dummy_concat2 = pd.concat(dummy2, axis=1)
                dummy_concat2 = dummy_concat2.loc[:,~dummy_concat.columns.duplicated()]
                dummy_concat2.drop(columns = disc_var, axis =1, inplace=True)
            except:
                continue
            
            dummyenc_df = pd.concat([df_input[cont_var], dummy_concat2, dummy_concat], axis=1)
            path = str(self.path) + "data_set_"+str(uuid4()) + str(".csv")
            dummyenc_df.to_csv(path,index=False)
            config = {}
            config["dataset_path"]= path
            config["preprocess_parameters"] = copy.deepcopy(self.config_before_encoding[l]["preprocess_parameters"])
            config["preprocess_parameters"]["encoding"] = "dummy_encoding"
            config["is_best"] = False
            self.config_dummy.append(config)
        self.config.extend(self.config_dummy)
        
        
    def get_datasets(self):
        self.withput_missing_value_in_target()
        self.numerical_var()
        self.categorical_var()
        self.continuous_var()
        self.discrete_var()
        self.knn()
        self.withput_imputation_column()
        self.withput_imputation_row()
        self.standardscaler()
        self.normalizer()
        self.minmaxscaler()
        self.ordinal_encoding()
        self.dummy_encoding()
        print(self.config)
    def info(self):
        self.withput_missing_value_in_target()
        self.column_unique()
        self.missing_value()
        print("Cat variables: ",self.cat_var)
        print("Numerical Variables: ",self.numeric_var)
        print("Discrete variables: ",self.disc_var)
        print("Continuous Variables :", self.cont_var)
    def dataset_numbers(self):
        self.get_datasets()
        print("Datasets by Imputation: ", len(self.configuration_knn))
        print("Dataset without imputaion: ", (len(self.config_wic)+len(self.config_wir)))
        print("Dataset by Standard Scalar: ", len(self.config_transform))
        print("Dataset by Normalizer: ",len(self.config_normalize))
        print("Dataset by Minmaxscalar: ", len(self.config_minmax))
        print("Dataset by Ordinal Encoding: ", len(self.config_encoding))
        print("Dataset by Dummy Encoding: ", len(self.config_dummy))
        


# In[ ]:


path=r"C:\Users\AH41803\OneDrive - Anthem\Documents\preprocess_sklearn_pipeline\dataset\2/"
pp=preprocessing(df,path)
pp.get_datasets()


# In[ ]:





# In[ ]:




