import numpy as np
import pandas as pd
from pygments.lexers import j
from scipy.stats import skew
import scipy.stats as st
import warnings
from pandas import factorize


class DataAnalyzer:
    def __init__(self, file_path=None, dataframe=None):
        """
        :param file_path: file_path to analyze
        :type file_path: str
        :param dataframe: dataframe to analyze
        :type dataframe: dataframe
        """

        self.file_path = file_path
        if not file_path:
            self.dataframe = dataframe
        else:
            self._load_data()
        self.no_of_rows = self.dataframe.shape[0]
        self.no_of_columns = self.dataframe.shape[1]
        self.target = self.dataframe.columns[-1]
        self.features = self.dataframe.columns[:-1]
        self.numerical_dataframe = None
        self.categorical_dataframe = None
        self.numerical_variables = []
        self.categorical_variables = []
        self.analysis = None

    def _load_data(self):
        """
        loads the dataframe from the given csv file.

        """
        self.dataframe = pd.read_csv(self.file_path, encoding='utf-8-sig')

    def _detect_column_type(self, row):
        """
        returns whether the column of the dataframe is numerical or categorical.

        :param row: row of the transformed dataframe
        :type row: series

        """
        if row['Columns'] in self.numerical_variables:
            return "Numerical"
        return "Categorical"

    def _set_column_type(self):
        """
        returns a list of data type of the columns of the dataframe.
        """
        self.analysis['column_type'] = self.analysis.apply(lambda row: self._detect_column_type(row), axis=1)

    def _set_numerical_categorical_columns(self):
        """
        Updates numerical and categorical details
        """

        numeric_var = [key for key in dict(self.dataframe.dtypes)
                       if dict(self.dataframe.dtypes)[key]
                       in ['float64', 'float32', 'int32', 'int64']]  # Numeric Variable

        """
        Identifying Discrete and Continuous Numerical variables
        """

        df_new = self.dataframe.dropna(axis=0, how='any')
        discrete_variables = []
        continuous_variables = []

        for i in numeric_var:
            flag = 0
            for k in df_new[str(i)]:
                if k - int(k) == 0:
                    flag += 1
            if flag == df_new[i].size and self.dataframe[str(i)].nunique() < 20:
                discrete_variables.append(i)
            else:
                continuous_variables.append(i)

        cat_var = [key for key in dict(self.dataframe.dtypes)
                   if dict(self.dataframe.dtypes)[key] in ['object']]  # Categorical Variable

        self.numerical_dataframe = self.dataframe[numeric_var]
        self.categorical_dataframe = self.dataframe[cat_var]
        self.numerical_variables = numeric_var
        self.categorical_variables = cat_var
        self.continuous_variables = continuous_variables
        self.discrete_variables = discrete_variables
        self.continuous_dataframe = self.dataframe[continuous_variables]
        self.discrete_dataframe = self.dataframe[discrete_variables]

    def _detect_target(self):
        if self.dataframe[self.target].dtype in ['object']:
            return ("categorical target")
        elif self.dataframe[self.target].dtype in ['float64', 'float32', 'int32', 'int64']:
            return ("numerical target")

    def _get_mean_of_columns(self):
        """
        calculates mean of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].mean())

    def _get_min_of_columns(self):
        """
        calculates minimum of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].min(), 2)

    def _get_max_of_columns(self):
        """
        calculates maximum of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].max(), 2)

    def _get_var_of_columns(self):
        """
        calculates variance of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].var(), 2)

    def _get_std_of_columns(self):
        """
        calculates standard deviation of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].std(), 2)

    def _get_median_of_columns(self):
        """
        calculates median of all columns in the dataframe.
        """

        return round(self.numerical_dataframe[list(self.numerical_dataframe.columns)].median(), 2)

    def _get_mode_of_columns(self):
        """
        calculates mode of all columns in the dataframe.
        """
        return self.dataframe.mode().iloc[0]

    def _get_missing_values(self):
        """
        calculates missing values in all columns of the dataframe.
        """
        missing_values = self.dataframe[list(self.dataframe.columns)].isna().sum()
        missing_values_percentage = self.dataframe[list(self.dataframe.columns)].isna().sum().div(self.no_of_rows)
        return missing_values, missing_values_percentage

    @staticmethod
    def _detect_unique_vals_of_columns(col):
        """
        calculates unique values in all columns of the dataframe.
        :param col: column of the dataframe
        :type col: series
        """
        return col.nunique()

    def _get_unique_values(self):
        """
        gives a list of unique values of each column in the dataframe.
        """

        unique_values_list = self.dataframe.apply(lambda col: DataAnalyzer._detect_unique_vals_of_columns(col), axis=0)
        return unique_values_list

    @staticmethod
    def _detect_skewness(col):
        """
        calculates skewness type of all columns in the dataframe.
        :param col: column of the dataframe
        :type col: series
        """

        skewness = skew(col)
        return float("{:.2f}".format(skewness))

    def _get_skewness(self):
        """
        gives a list of skewness values of all columns in the dataframe.
        """
        skewness_values = self.numerical_dataframe.apply(lambda col: DataAnalyzer._detect_skewness(col), axis=0)
        return skewness_values

    @staticmethod
    def _detect_outliers(col, total_rows):
        """
        detects outliers of all columns in the dataframe.
        :param col: column of the dataframe
        :type col: series
        """

        quartile1, quartile3 = np.percentile(col, [25, 75])
        iqr = quartile3 - quartile1
        lower_bound = quartile1 - (1.5 * iqr)
        upper_bound = quartile3 + (1.5 * iqr)
        outliers = ((col < lower_bound) | (col > upper_bound)).sum()
        outlier_percentage = (outliers / total_rows) * 100
        return pd.Series((outliers, iqr, outlier_percentage))

    def _get_outliers(self):
        """
        gives a list of outliers of all columns in the dataframe.
        """

        series_res = self.numerical_dataframe.apply(lambda col: DataAnalyzer._detect_outliers(col, self.no_of_rows),
                                                    axis=0, result_type='expand')
        return series_res.iloc[0], series_res.iloc[1], series_res.iloc[2]

    def chi_square_test(self, col):
        alpha = 0.1
        dependent_cat = ''
        for i in range(len(self.categorical_variables)):
            myfield1 = col
            myfield2 = self.dataframe[self.categorical_variables[i]]
            mycrosstable = pd.crosstab(myfield1, myfield2)
            stat, p, dof, expected = st.chi2_contingency(mycrosstable)
            if p > (1 - alpha) and p != 1:
                dependent_cat = dependent_cat + " " + str(self.categorical_variables[i])
        if dependent_cat == '':
            dependent_cat = -1

        return (dependent_cat)

    def _get_correlation_categoriacal(self):
        categorical_correlated = self.categorical_dataframe.apply(lambda col: self.chi_square_test(col),
                                                                  axis=0)
        return categorical_correlated

    def correlation_matrix(self):
        return self.numerical_dataframe.corr(method='pearson')

    def numerical_correlation(self, col):
        correlated = ''
        for i in range(len(self.numerical_variables)):
            correlation = col.corr(self.dataframe[self.numerical_variables[i]])
            if (correlation > 0.8 or correlation < (-0.8)) and col.name != self.numerical_variables[i]:
                correlated = correlated + " " + str(self.numerical_variables[i])
        if correlated == '':
            correlated = -1

        return correlated

    def _get_correlation_numerical(self):
        numerical_correlated = self.numerical_dataframe.apply(lambda col: DataAnalyzer.numerical_correlation(self, col),
                                                              axis=0)
        return numerical_correlated

    def _get_correlation_with_target(self):
        """
        computes information gain of each feature
        :return: series
        """
        item_values = []
        for feature in self.features:
            if self._detect_target() == "categorical target":
                if feature in self.numerical_variables:
                    labels, categories = factorize(self.dataframe[self.target])
                    s1 = pd.Series(labels)
                    item_values.append(abs(self.dataframe[feature].corr(s1)))
                else:
                    labels, categories = factorize(self.dataframe[self.target])
                    s1 = pd.Series(labels)
                    labels2, categories2 = factorize(self.dataframe[feature])
                    s2 = pd.Series(labels2)
                    item_values.append(abs(s2.corr(s1)))
            else:
                if feature in self.numerical_variables:
                    item_values.append(abs(self.dataframe[feature].corr(self.dataframe[self.target])))
                else:
                    labels, categories = factorize(self.dataframe[feature])
                    s2 = pd.Series(labels)
                    item_values.append(abs((self.dataframe[self.target]).corr(s2)))

        return pd.Series(item_values, index=self.features)

    def _get_best_fit_distribution_of_columns(self):
        best_distribution_list = []
        best_param_list = []
        df_new = self.numerical_dataframe.fillna(self.dataframe.mean())

        for i in self.dataframe.columns:
            if i in self.continuous_variables:
                best_distribution, best_param = self._get_best_fit_continuous(df_new[str(i)])
                best_distribution_list.append(best_distribution)
                best_param_list.append(best_param)

            else:
                best_distribution_list.append("NA")
                best_param_list.append("NA")

        best_distribution_l = pd.Series(data=best_distribution_list, index=self.dataframe.columns)
        best_param_l = pd.Series(data=best_param_list, index=self.dataframe.columns)

        return best_distribution_l, best_param_l

    def _get_best_fit_continuous(self, df):

        length = self.no_of_rows
        y, x = np.histogram(df, bins=int(np.log2(length)) + 1, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        DISTRIBUTIONS = ['alpha', 'beta', 'cauchy', 'chi', 'chi2', 'expon', 'exponnorm', 'gennorm', 'genexpon', 'gamma',
                         'laplace', 'lognorm', 'norm', 'pareto', 'powernorm', 'uniform', 'weibull_min', 'weibull_max']

        best_distribution = 'norm'
        best_params = (0, 1)
        best_sse = np.inf

        for distribution in DISTRIBUTIONS:
            dist = getattr(st, distribution)
            # try to fit the distribution
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')

                    # fit dist to data
                    params = dist.fit(df)

                    # separate parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]

                    # Calculate fitted PDF and error with fit in the distribution
                    pdf1 = dist.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf1, 2))

                    # identification if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution
                        best_params = params
                        best_sse = sse

            except Exception:
                pass
        return best_distribution, best_params

    def _run_analysis(self):
        """ gets all statistics of all columns from loaded dataframe and creates a new dataframe"""
        self._set_numerical_categorical_columns()
        unique_values = self._get_unique_values()
        maximum_values = self._get_max_of_columns()
        mean_values = self._get_mean_of_columns()
        minimum_values = self._get_min_of_columns()
        variance_values = self._get_var_of_columns()
        std_values = self._get_std_of_columns()
        median_values = self._get_median_of_columns()
        categorical_correlation = self._get_correlation_categoriacal()
        numerical_correlation = self._get_correlation_numerical()
        skewness_values = self._get_skewness()
        outlier_values, iqr_values, outlier_percentage = self._get_outliers()
        mode_values = self._get_mode_of_columns()
        missing_values, missing_values_percentage = self._get_missing_values()
        best_distribution, best_param = self._get_best_fit_distribution_of_columns()
        correlation_with_target = self._get_correlation_with_target()

        # TODO: Re-enable Correlation when we have a faster calculation.
        # Currently the Information Gain based method takes 10 - 20 mins.
        # Once we have a better solution we'll update it.
        # correlation = self._get_correlation_with_target()

        data_analysed = {
            'unique': unique_values,
            'max': maximum_values,
            'mean': mean_values,
            'min': minimum_values,
            'var': variance_values,
            'std_deviation': std_values,
            'median': median_values,
            'skew': skewness_values,
            'mode': mode_values,
            'iqr_value': iqr_values,
            'no_of_outliers': outlier_values,
            'outlier_percentage': outlier_percentage,
            'no_of_null_values': missing_values,
            'null_values_percentage': missing_values_percentage,
            'categorical_variable_correlation': categorical_correlation,
            'numerical_variable_correlation': numerical_correlation,
            'best_distribution': best_distribution,
            'best_parameter': best_param,
            'correlation_with_target': correlation_with_target
        }
        self.analysis = pd.DataFrame(data_analysed).fillna(-1)
        self.analysis.categorical_variable_correlation = self.analysis.categorical_variable_correlation.astype(str)
        self.analysis.numerical_variable_correlation = self.analysis.numerical_variable_correlation.astype(str)
        self.analysis['total_instances'] = self.no_of_rows
        self.analysis['suggestions'] = 'This feature is good to go'

    def get_analysis(self):
        """
        gets all statistics of all columns from loaded dataframe and creates a new dataframe
        :return dataframe
        """
        self._run_analysis()
        self.analysis.reset_index(level=0, inplace=True)
        self.analysis.rename(columns={'index': 'columns'}, inplace=True)
        self.suggestions()
        return self.analysis

    def suggestions(self):
        dist = ['alpha', 'beta', 'cauchy', 'chi', 'chi2', 'expon', 'genexpon', 'gamma', 'laplace', 'pareto', 'uniform',
                'weibull_min', 'weibull_max']
        for index, row in self.analysis.iterrows():
            unique_value_count = self.dataframe[row['columns']].value_counts(normalize=True)
            x = unique_value_count.max()-unique_value_count.min()
            y = 1.25*(1/len(unique_value_count))
            if row['outlier_percentage'] > 15 and row['columns']!= self.target:
                self.analysis.loc[index,'suggestions'] = 'High Outlier Percentage'
            elif row['unique'] == 1:
                self.analysis.loc[index,'suggestions'] = 'Only One Unique value in the feature'
            elif (row['var'] > 5000) and (row['correlation_with_target'] < 0.05) and (row['columns'] != self.target):
                self.analysis.loc[index, 'suggestions'] = 'Noisy Feature'
            elif len(str(row['numerical_variable_correlation']).split(" ")) != 1:
                self.analysis.loc[index,'suggestions'] = "The feature is highly related to other features:" + (",".join(str(row['numerical_variable_correlation']).split(" ")))
            elif len(str(row['categorical_variable_correlation']).split(" ")) != 1:
                self.analysis.loc[index,'suggestions'] = "The feature is highly related to other features:" + (",".join(str(row['categorical_variable_correlation']).split(" ")))
            elif row['null_values_percentage'] > 30:
                self.analysis.loc[index,'suggestions'] = 'High percentage of null values'
            elif (x > y) and (row['columns'] in self.categorical_variables) :
                self.analysis.loc[index, 'suggestions'] = 'Not enough samples of all categories'
            elif (x > y) and (row['columns'] == self.target) :
                self.analysis.loc[index, 'suggestions'] = 'Class Imbalance'
            elif row['skew'] > 1 :
                self.analysis.loc[index,'suggestions'] = 'Highly positive skewed'
            elif row['skew'] < -1:
                self.analysis.loc[index,'suggestions'] = 'Highly negative skewed'
            elif row['best_distribution'] in dist:
                self.analysis.loc[index,'suggestions'] = 'This feature can be standardized'

        return self.analysis


#da = DataAnalyzer(file_path="C:/Users/AH41803/Downloads/heart_failure_clinical_records_dataset.csv")
#res = da.get_analysis()
