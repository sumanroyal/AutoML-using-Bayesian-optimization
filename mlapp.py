import streamlit as st
import pandas as pd
from data_analyzer import DataAnalyzer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
from feature_engine.outliers import Winsorizer
from feature_engine.imputation import CategoricalImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from feature_engine.encoding import OrdinalEncoder
from feature_engine.encoding import OneHotEncoder
from PIL import Image
from pipeline import automl_pipeline
from bayes import automl_bayes_pipeline
from feature_engine.imputation import DropMissingData
from feature_engine.imputation import MeanMedianImputer
import time

st.title('AutoML')
image = Image.open('logo1.jpg')
st.image(image,use_column_width=True)


def auto_ml():
    activities = ['EDA', 'Review', 'Visualisation','Model-For Data Scientists', 'AutoML','AutoML-Bayes Optimization','About us']
    option = st.sidebar.selectbox('Selection option:', activities)

    if option == 'EDA':
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        st.success("Data successfully loaded")

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))

            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox('Display Correlation of data various columns'):
                st.write(df.corr())

    elif option == "Review":
        st.subheader("Review Uploaded Data")
        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        st.success("Data successfully loaded")

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            if st.checkbox("Review data"):
                da = DataAnalyzer(dataframe=df)
                analysis_df = da.get_analysis()
                streamlit_df = analysis_df.drop(['best_parameter', 'mode'], axis=1)
                st.subheader("Review Metrics")
                st.dataframe(streamlit_df.iloc[:, ~streamlit_df.columns.isin(['suggestions'])])
                st.subheader("Review Suggestions")
                st.dataframe(streamlit_df[['columns', 'suggestions']])

    elif option == 'Visualisation':
        st.subheader("Data Visualisation")

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        st.success("Data successfully loaded")
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple columns to plot'):
                selected_columns = st.multiselect('Select your preferred columns', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

                if st.checkbox('Display Heatmap'):
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.write(sns.heatmap(df1.corr(), vmin=-1,vmax=1, square=True, annot=True, cmap='viridis'))
                    st.pyplot()
                if st.checkbox('Display Pairplot'):
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.write(sns.pairplot(df1, diag_kind='kde'))
                    st.pyplot()
                if st.checkbox('Boxplot'):
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    df1.boxplot()
                    st.pyplot()
                if st.checkbox('Display Pie Chart'):
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    all_columns = df.columns.to_list()
                    pie_columns = st.selectbox("select column to display", all_columns)
                    pieChart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                    st.write(pieChart)
                    st.pyplot()



    elif option == 'Model-For Data Scientists':
        st.subheader("Model Building")

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        st.success("Data successfully loaded")
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            seed = st.sidebar.slider('Seed', 1, 200)

            if st.checkbox('Select Multiple columns'):
                if st.checkbox('Select All Columns', value=True):
                    df1=df
                    X = df1.iloc[:, 0:-1]
                    y = df1.iloc[:, -1]
                else:
                    new_data = st.multiselect("Select your preferred columns. NB: Let your target variable be the last column to be selected",
                                              df.columns)
                    df1 = df[new_data]
                    X = df1.iloc[:, 0:-1]
                    y = df1.iloc[:, -1]
                st.dataframe(df1)

                # Dividing my data into X and y variables


                categorical = [var for var in X.columns if df[var].dtype == 'O']
                discrete = [var for var in X.columns if df[var].dtype != 'O' and (df[var].nunique()) <= 10]
                continuous = [var for var in X.columns if df[var].dtype != 'O' and var not in discrete]
                st.write("Categorical Variables:",categorical)
                st.write('Continuous Variables:',continuous)
                st.write('Discrete Variables',discrete)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
                if st.checkbox('Outlier Capping'):
                    if len(continuous)!=0:
                        capper = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=continuous)
                        # fit the capper
                        capper.fit(X_train)

                        # transform the data
                        X_train = capper.transform(X_train)
                        X_test = capper.transform(X_test)

                if st.checkbox('Missing Value Handling'):
                    imputer_name=st.selectbox('Select the method of Handling',('Drop Missing Data',
                                                                                 'Imputation'))
                    if imputer_name=='Drop Missing Data':
                        missingdata_imputer = DropMissingData()
                        missingdata_imputer.fit(X_train)

                        # transform the data
                        X_train = missingdata_imputer.transform(X_train)
                        X_test = missingdata_imputer.transform(X_test)

                    if imputer_name == 'Imputation':
                        if len(continuous)!=0:
                            mean_imputer = MeanMedianImputer(imputation_method='mean', variables=continuous)

                            # fit the imputer
                            mean_imputer.fit(X_train)

                            # transform the data
                            X_train = mean_imputer.transform(X_train)
                            X_test = mean_imputer.transform(X_test)
                        if len(discrete)!=0:
                            median_imputer = MeanMedianImputer(imputation_method='median',
                                                               variables=discrete)

                            # fit the imputer
                            median_imputer.fit(X_train)

                            # transform the data
                            X_train = median_imputer.transform(X_train)
                            X_test = median_imputer.transform(X_test)

                        if len(categorical)!=0:
                            # set up the imputer
                            imputer = CategoricalImputer(imputation_method='frequent',variables=categorical)

                            # fit the imputer
                            imputer.fit(X_train)

                            # transform the data
                            X_train = imputer.transform(X_train)
                            X_test = imputer.transform(X_test)
                if st.checkbox('Normalisation'):
                    if len(continuous)!=0:
                        normaliser=SklearnTransformerWrapper(transformer=MinMaxScaler(),variables=continuous)
                        normaliser.fit(X_train)
                        X_train=normaliser.transform(X_train)
                        X_test=normaliser.transform(X_test)

                if st.checkbox('Standardisation'):
                    if len(continuous)!=0:
                        standardiser=SklearnTransformerWrapper(transformer=StandardScaler(),variables=continuous)
                        standardiser.fit(X_train)
                        X_train=standardiser.transform(X_train)
                        X_test=standardiser.transform(X_test)
                if st.checkbox('Categorical Encoding',value=True):
                    encoder_name = st.selectbox('Select the method of Encoding', ('OneHot',
                                                                                  'Label'))
                    if len(categorical)!=0:
                        if encoder_name=='OneHot':
                            encoder = OneHotEncoder(variables=categorical,
                                                    drop_last=True)

                            # fit the encoder
                            encoder.fit(X_train)

                            # transform the data
                            X_train = encoder.transform(X_train)
                            X_test = encoder.transform(X_test)
                        if encoder_name == 'Label':
                            encoder = OrdinalEncoder(encoding_method='arbitrary', variables=categorical)

                            # fit the encoder
                            encoder.fit(X_train)

                            # transform the data
                            X_train = encoder.transform(X_train)
                            X_test = encoder.transform(X_test)


                st.dataframe(X_train)




                classifier_name = st.sidebar.selectbox('Select your preferred classifier:',
                                                       ('KNN', 'SVM', 'LR', 'naive_bayes', 'decision tree'))

                def add_parameter(name_of_clf):
                    params = dict()
                    if name_of_clf == 'SVM':
                        C = st.sidebar.slider('C', 0.01, 15.0)
                        params['C'] = C
                    elif name_of_clf == 'KNN':
                        K = st.sidebar.slider('K', 1, 15)
                        params['K'] = K
                    elif name_of_clf == 'decision tree':
                        D = st.sidebar.slider('Max Depth of Trees', 3, 15)
                        params['Max Depth of Trees'] = D
                    return params

                # calling the function

                params = add_parameter(classifier_name)

                # defing a function for our classifier

                def get_classifier(name_of_clf, params):
                    clf = None
                    if name_of_clf == 'SVM':
                        clf = SVC(C=params['C'])
                    elif name_of_clf == 'KNN':
                        clf = KNeighborsClassifier(n_neighbors=params['K'])
                    elif name_of_clf == 'LR':
                        clf = LogisticRegression()
                    elif name_of_clf == 'naive_bayes':
                        clf = GaussianNB()
                    elif name_of_clf == 'decision tree':
                        clf = DecisionTreeClassifier(max_depth=params['Max Depth of Trees'])
                    else:
                        st.warning('Select your choice of algorithm')

                    return clf

                clf = get_classifier(classifier_name, params)



                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                st.write('Predictions:', y_pred)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()

                st.write('Nmae of classifier:', classifier_name)
                st.write('Accuracy', accuracy)
                st.write('Report')
                st.dataframe(report_df)


    elif option == "AutoML":
        st.markdown(' Upload the file with target column at the end. Results will be mailed to you')

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        st.success("Data successfully loaded")
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            scoring_method = st.sidebar.selectbox('Select your preferred scoring metrics:',
                                                   ('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'))

            if st.checkbox("Perform AutoML"):
                start = time.time()
                st.write("Fetching Report .....................................")
                params, report ,con_mat = automl_pipeline(df,scoring_method)
                end = time.time()
                t = end-start
                st.write('Time taken:', t)
                st.dataframe(report)
                st.write('Confusion Matrix')
                st.dataframe(con_mat)
                st.write('Best Parameters', params)


    elif option == "AutoML-Bayes Optimization":
        st.markdown(' Upload the file with target column at the end. Results will be mailed to you')

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        st.success("Data successfully loaded")
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            scoring_method = st.sidebar.selectbox('Select your preferred scoring metrics:',
                                                   ('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'))

            if st.checkbox("Perform AutoML"):
                start = time.time()
                st.write("Fetching Report .....................................")
                params, report ,con_mat = automl_bayes_pipeline(df,scoring_method)
                end = time.time()
                t = end - start
                st.write('Time taken:', t)
                st.dataframe(report)
                st.write('Confusion Matrix')
                st.dataframe(con_mat)
                st.write('Best Parameters', params)




    elif option == 'About us':

        st.markdown(
            'This is developed by Suman Roy.'
            )
        st.markdown(
            """<a href="https://github.com/sumanroyal/AutoML-using-Bayesian-optimization">Github</a>""", unsafe_allow_html=True,
        )

        st.balloons()


if __name__ == '__main__':
    auto_ml()