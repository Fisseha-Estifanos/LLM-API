"""
A script to clean data.
"""

# imports
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import logging
from sklearn import preprocessing


class dataCleaner():
    """
    A data cleaner class.
    """
    def __init__(self, fromThe: str) -> None:
        """
        The data cleaner initializer

        Parameters
        =--------=
        fromThe: string
            The file importing the data cleaner

        Returns
        =-----=
        None: nothing
            This will return nothing, it just sets up the data cleaner
            script.
        """
        try:
            # setting up logger
            self.logger = self.setup_logger('../logs/cleaner_root.log')
            self.logger.info('\n    #####-->    Data cleaner logger for ' +
                             f'{fromThe}    <--#####\n')
            print('Data cleaner in action')
        except Exception as e:
            print(e)

    def setup_logger(self, log_path: str) -> logging.Logger:
        """
        A function to set up logging

        Parameters
        =--------=
        log_path: string
            The path of the file handler for the logger

        Returns
        =-----=
        logger: logger
            The final logger that has been setup up
        """
        try:
            # getting the log path
            log_path = log_path

            # adding logger to the script
            logger = logging.getLogger(__name__)
            print(f'--> {logger}')
            # setting the log level to info
            logger.setLevel(logging.DEBUG)
            # setting up file handler
            file_handler = logging.FileHandler(log_path)

            # setting up formatter
            formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(funcName)s] [%(levelname)s] " +
                "--> [%(message)s]")

            # setting up file handler and formatter
            file_handler.setFormatter(formatter)
            # adding file handler
            logger.addHandler(file_handler)
            print(f'logger {logger} created at path: {log_path}')
        except Exception as e:
            logger.error(e, exec_info=True)
            print(e)
        finally:
            # return the logger object
            return logger

    def remove_unwanted_cols(self, df: pd.DataFrame,
                             cols: list) -> pd.DataFrame:
        """
        A function to remove unwanted features from a DataFrame

        Parameters
        =--------=
        df: pandas dataframe
            The data frame containing all the data
        cols: list
            The unwanted features lists

        Returns
        =-----=
        df
            The dataframe rid of the unwanted cols
        """
        try:
            for col in cols:
                df = df[df.columns.drop(list(df.filter(regex = col)))]
                self.logger.info(f'feature: {col} removed successfully')
                print(f'feature: {col} removed successfully')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df        

    def percent_missing(self, df: pd.DataFrame) -> None:
        """
        A function telling how many missing values exist or better still
        what is the % of missing values in the dataset?

        Parameters
        =--------=
        df: pandas dataframe
            The data frame to calculate the missing values from

        Returns
        =-----=
        None: nothing
            Just prints the missing value percentage
        """
        try:
            # Calculate total number of cells in dataframe
            totalCells = np.product(df.shape)

            # Count number of missing values per feature
            missingCount = df.isnull().sum()

            # Calculate total number of missing values
            totalMissing = missingCount.sum()

            # Calculate percentage of missing values
            print(f"The dataset contains {round(((totalMissing/totalCells)*100), 10)} % missing values")
            self.logger.info(f"The dataset contains {round(((totalMissing/totalCells)*100), 10)} % missing values")
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)

    # TODO: compare this fill method with others
    def fillWithMedian(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        A function that fills null values with their corresponding median
        values

        Parameters
        =--------=
        df: pandas data frame
            The data frame with the null values
        cols: list
            The list of features to be filled with median values

        Returns
        =-----=
        df: pandas data frame
            The data frame with the null values replace with their
            corresponding median values
        """
        try:
            print(f'features to be filled with median values: {cols}')
            self.logger.info(f'features to be filled with median values: {cols}')
            df[cols] = df[cols].fillna(df[cols].median())
            self.logger.info(f'features: {cols} filled with median successfully')
            print(f'features: {cols} filled with median successfully')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df

    # TODO: compare this fill method with others
    # TODO: also group these methods together, write them one after the other
    # instead of putting them in a scattered manner all over this script
    def fillWithMean(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        A function that fills null values with their corresponding mean
        values

        Parameters
        =--------=
        df: pandas data frame
            The data frame with the null values
        cols: list
            The list of features to be filled with mean values

        Returns
        =-----=
        df: pandas data frame
            The data frame with the null values replace with their
            corresponding mean values
        """
        try:
            self.logger.info(f'features to be filled with mean values: {cols}')
            print(f'features to be filled with mean values: {cols}')
            df[cols] = df[cols].fillna(df[cols].mean())
            self.logger.info(f'cols: {cols} filled with mean successfully')
            print(f'cols: {cols} filled with mean successfully')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df

    # TODO : compare the two outlier fixers
    def fix_outlier(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        A function to fix outliers with median

        Parameters
        =--------=
        df: pandas data frame
            The data frame containing the outlier features
        column: str
            The string name of the feature with the outlier problem

        Returns
        =-----=
        df: pandas data frame
            The fixed data frame
        """
        try:
            self.logger.info(f'feature to be filled with median values: {column}')
            print(f'feature to be filled with median values: {column}')
            df[column] = np.where(df[column] > df[column].quantile(0.95),
                                  df[column].median(), df[column])
            self.logger.info(f'feature: {column} outlier fixed successfully')
            print(f'feature: {column} outlier fixed successfully')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df[column]

    def replace_outlier_with_median(self, dataFrame: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        A function to fix outliers with median
        
        Parameters
        =--------=
        df: pandas data frame
            The data frame containing the outlier features
        feature: str
            The string name of the feature with the outlier problem

        Returns
        =-----=
        dataFrame: pandas data frame
            The fixed data frame
        """
        try:
            Q1 = dataFrame[feature].quantile(0.25)
            Q3 = dataFrame[feature].quantile(0.75)
            median = dataFrame[feature].quantile(0.50)

            IQR = Q3 - Q1

            upper_whisker = Q3 + (1.5 * IQR)
            lower_whisker = Q1 - (1.5 * IQR)

            dataFrame[feature] = np.where(
                dataFrame[feature] > upper_whisker, median, dataFrame[feature])
            self.logger.info(f'feature: {feature} outlier values greater than: {upper_whisker} fixed successfully with the median value of: {median}')
            print(f'feature: {feature} outlier values greater than: {upper_whisker} fixed successfully with the median value of: {median}')
            dataFrame[feature] = np.where(
                dataFrame[feature] < lower_whisker, median, dataFrame[feature])
            self.logger.info(f'feature: {feature} outlier values less than: {lower_whisker} fixed successfully with the median value of: {median}')
            print(f'feature: {feature} outlier values less than: {lower_whisker} fixed successfully with the median value of: {median}')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return dataFrame

    def choose_k_means(self, df: pd.DataFrame, num: int):
        """
        A function to choose the optimal k means cluster

        Parameters
        =--------=
        df: pandas data frame
            The data frame that holds all the values
        num: integer
            The x scale

        Returns
        =-----=
        distortions and inertias
        """
        try:
            distortions = []
            inertias = []
            K = range(1, num)
            for k in K:
                k_means = KMeans(n_clusters=k, random_state=777).fit(df)
                distortions.append(sum(
                    np.min(cdist(df, k_means.cluster_centers_, 'euclidean'),
                           axis=1)) / df.shape[0])
                inertias.append(k_means.inertia_)
            self.logger.info(f'distortion: {distortions} and inertia:' +
                             f'{inertias} calculated for {num} number of'
                             'clusters successfully')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return (distortions, inertias)

    def computeBasicAnalysisOnClusters(self, df: pd.DataFrame,
                                       cluster_col: str, cluster_size: int,
                                       cols: list) -> None:
        """
        A function that gives some basic description of the 3 clusters

        Parameters
        =--------=
        df: pandas data frame
            The main data frame containing all the data
        cluster_col: str
            The feature name holding the cluster values
        cluster_size: integer
            The number of total cluster groups
        cols: list
            The feature list on which to provide description

        Returns
        =-----=
        None: nothing
            This function only prints out information
        """
        try:
            i = 0
            for i in range(cluster_size):
                cluster = df[df[cluster_col] == i]
                print("Cluster " + (i+1) * "I")
                print(cluster[cols].describe())
                print("\n")
            self.logger.info(f'basic analysis on {cluster_size} clusters' +
                             'computed successfully')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)



    # new additions
    # TODO: add try, except finally _ DONE
    # TODO: add comment _ DONE
    # TODO: add logger _ DONE
    # TODO: PEP8

    # TODO: compare this fill method with others
    def fix_missing_ffill(self, df: pd.DataFrame, cols: list) -> None:
        """
        A function to fill missing values with the ffill method

        Parameters
        =--------=
        df: pandas dataframe
            The main dataframe
        cols: list
            A list containing the missing values

        Returns
        =-----=
        None: nothing
            Just fills the missing values
        """
        try:
            for col in cols:
                old = df[col].isna().sum()
                df[col] = df[col].fillna(method='ffill')
                new = df[col].isna().sum()
                if new == 0:
                    print(f"{old} missing values in the feature {col} have been replaced \
                        using the forward fill method.")
                    self.logger.info(f"{old} missing values in the feature {col} have been replaced \
                        using the forward fill method.")
                else:
                    count = old - new
                    print(f"{count} missing values in the feature {col} have been replaced \
                        using the forward fill method. {new} missing values that couldn't be \
                        imputed still remain in the feature {col}.")
                    self.logger.info(f"{count} missing values in the feature {col} have been replaced \
                        using the forward fill method. {new} missing values that couldn't be \
                        imputed still remain in the features {col}.")
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)

    # TODO: compare this fill method with others
    def fix_missing_bfill(self, df: pd.DataFrame, cols: list) -> None:
        """
        A function to fill missing values with the bfill method

        Parameters
        =--------=
        df: pandas dataframe
            The main dataframe
        cols: list
            A list containing the missing values

        Returns
        =-----=
        None: nothing
            Just fills the missing values
        """
        try:
            for col in cols:
                old = df[col].isna().sum()
                df[col] = df[col].fillna(method='bfill')
                new = df[col].isna().sum()
                if new == 0:
                    self.logger.info(f"{old} missing values in the feature {col} have been replaced \
                        using the backward fill method.")
                    print(f"{old} missing values in the feature {col} have been replaced \
                        using the backward fill method.")
                else:
                    count = old - new
                    self.logger.info(f"{count} missing values in the feature {col} have been replaced \
                        using the backward fill method. {new} missing values that couldn't be \
                        imputed still remain in the feature {col}.")
                    print(f"{count} missing values in the feature {col} have been replaced \
                        using the backward fill method. {new} missing values that couldn't be \
                        imputed still remain in the feature {col}.")
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)

    def missing_values_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A function to calculate missing values by features

        Parameters
        =--------=
        df: pandas dataframe
            The main dataframe
        
        Returns
        =-----=
        mis_val_table_ren_columns: pandas data frame
            The data frame containing missing value information
        """
        try:
            # Total missing values
            mis_val = df.isnull().sum()

            # Percentage of missing values
            mis_val_percent = 100 * mis_val / len(df)

            # dtype of missing values
            mis_val_dtype = df.dtypes

            # Make a table with the results
            mis_val_table = pd.concat([mis_val, mis_val_percent, mis_val_dtype],
                                    axis=1)

            # Rename the features
            mis_val_table_ren_columns = mis_val_table.rename(
            columns = {0 : 'Missing Values', 1 : '% of Total Values', 2: 'Dtype'})

            # Sort the table by percentage of missing descending and remove features with no missing values
            mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:,0] != 0].sort_values(
            '% of Total Values', ascending=False).round(2)

            # Print some summary information
            print ("Your selected dataframe has " + str(df.shape[1]) + " features.\n"
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                " features that have missing values.")
            self.logger.info("Your selected dataframe has " + str(df.shape[1]) + " features.\n"
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                " features that have missing values.")
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            if mis_val_table_ren_columns.shape[0] == 0:
                return
            # Return the dataframe with missing information
            return mis_val_table_ren_columns

    # TODO: compare this fill method with others
    def fix_missing_value(self, df: pd.DataFrame, cols: list, value: int) -> None:
        """
        A function to fix missing values by a given value

        Parameters
        =--------=
        df: pandas dataframe
            The main dataframe
        cols: list
            List of features containing the names of the missing values
        value: integer
            The value to fill the missing values with
        
        Returns
        =-----=
        None: noting
            Just fills the missing value with a given value
        """
        try:
            for col in cols:
                count = df[col].isna().sum()
                df[col] = df[col].fillna(value)
                if type(value) == 'str':
                    self.logger.info(f"{count} missing values in the feature {col} have been replaced by \'{value}\'.")
                    print(f"{count} missing values in the feature {col} have been replaced by \'{value}\'.")
                else:
                    self.logger.info(f"{count} missing values in the feature {col} have been replaced by {value}.")
                    print(f"{count} missing values in the feature {col} have been replaced by {value}.")
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)

    # TODO: compare this fill method with others
    # TODO: test this method it has not been tested
    def fill_missing_rolling(self, df: pd.DataFrame, cols: list, window: int=5, min_period=2) -> pd.DataFrame:
        """
        A function to fill missing values using the rolling method

        Parameters
        =--------=
        df: pandas data frame
            The main data frame
        cols: list
            List of features with missing values
        window: integer
            The window to calculate rolling method on

        min_period: integer
            The minimum value to use for the rolling method

        Returns
        =-----=
        df: pandas data frame
            The same data frame with the missing values filled using the
            rolling method
        """
        try:
            for col in cols:
                df[col] = df[col].fillna(df[col].rolling(window=window, 
                                         min_periods=min_period).mean())
                self.logger.info(f'column: {col} filled with the rolling ' + 
                                 f'method, window:{window}, min_periods: ' +
                                 f'{min_period}')
                print(f'column: {col} filled with the rolling method, ' +
                      f'window:{window}, min_periods: {min_period}')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df
        
 
    def convert_to_string(self, df: pd.DataFrame, columns: list) -> pd.DataFrame :
        """
        A function to convert features to string data type

        Parameters
        =--------=
        df: pandas dataframe
            The main dataframe
        columns: list
            List of features to be converted to string data types
        
        Returns
        =-----=
        df: pandas data frame
            The converted data frame
        """
        try: 
            for col in columns:
                df[col] = df[col].astype("string")
                self.logger.info(f'feature: {col} converted to string data type format')
                print(f'feature: {col} converted to string data type format')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df

    def convert_to_numeric(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        A function to convert features to numeric data type

        Parameters
        =--------=
        df: pandas dataframe
            The main dataframe
        columns: list
            List of features to be converted to numeric data types
        
        Returns
        =-----=
        df: pandas data frame
            The converted data frame
        """
        try:
            for col in columns:
                df[col] = pd.to_numeric(df[col])
                self.logger.info(f'feature: {col} converted to numeric data type format')
                print(f'feature: {col} converted to numeric data type format')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df

    def convert_to_int(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        A function to convert features to integer data type

        Parameters
        =--------=
        df: pandas dataframe
            The main dataframe
        columns: list
            List of features to be converted to integer data types
        
        Returns
        =-----=
        df: pandas data frame
            The converted data frame
        """
        try:
            for col in columns:
                df[col] = df[col].astype("int64")
                self.logger.info(f'feature: {col} converted to integer data type format')
                print(f'feature: {col} converted to integer data type format')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df

    def convert_to_datetime(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        A function to convert features to datetime data type

        Parameters
        =--------=
        df: pandas dataframe
            The main dataframe
        columns: list
            List of features to be converted to datetime data types
        
        Returns
        =-----=
        df: pandas data frame
            The converted data frame
        """
        try:
            for col in columns:
                df[col] = pd.to_datetime(df[col], errors='raise')
                self.logger.info(f'feature: {col} successfully changed to datetime')
                print(f'feature: {col} successfully changed to datetime')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df

    def multiply_by_factor(self, df: pd.DataFrame, columns: list, factor: float) -> pd.DataFrame:
        """
        A function that multiplies a features by a given factor

        Parameters
        =--------=
        df: pandas dataframe
            The main dataframe
        columns: list
            List of features to be multiplied by a factor
        factor: float
            The multiplying factor
        
        Returns
        =-----=
        df: pandas data frame
            The multiplied data frame
        """
        try:
            for col in columns:
                df[col] = df[col] * factor
                self.logger.info(f'feature: {col} multiplied by a factor of: {factor}')
                print(f'feature: {col} multiplied by a factor of: {factor}')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return df

    def show_cols_mixed_dtypes(self, df: pd.DataFrame) -> None:
        """
        A function to show mixed data types

        Parameters
        =--------=
        df: pandas data frame
            The main data frame

        Returns
        =-----=
        None: nothing
            Just prints the mixed data types
        """
        try:
            mixed_dtypes = {'Column': [], 'Data type': []}
            for col in df.columns:
                dtype = pd.api.types.infer_dtype(df[col])
                if dtype.startswith("mixed"):
                    mixed_dtypes['Column'].append(col)
                    mixed_dtypes['Data type'].append(dtype)
            if len(mixed_dtypes['Column']) == 0:
                self.logger.info('None of the features contain mixed types.')
                print('None of the features contain mixed types.')
            else:
                self.logger.info(pd.DataFrame(mixed_dtypes))
                print(pd.DataFrame(mixed_dtypes))
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)

    def drop_duplicates(self, df: pd.DataFrame) -> None:
        """
        A function to drop duplicates

        Parameters
        =--------=
        df: pandas data frame
            The main data frame

        Returns
        =-----=
        None: nothing
            Just drops duplicates from the data set
        """
        try:
            old = df.shape[0]
            df.drop_duplicates(inplace=True)
            new = df.shape[0]
            count = old - new
            if (count == 0):
                self.logger.info("No duplicate rows were found.")
                print("No duplicate rows were found.")
            else:
                self.logger.info(f"{count} duplicate rows were found and removed.")
                print(f"{count} duplicate rows were found and removed.")
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
    
    def getMonth(self, month_list: list, index: int) -> int:
        """
        A function to return the index of a given month

        Parameters
        =--------=
        month_lits: list
            List of months
        index: int
            The index of the required grouping
        Returns
        =-----=
        months.index: int
            The index of the given month
        """
        try:
            months = ['0', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
            month_list = month_list.split(',')
            month = month_list[index]
            self.logger.info(f'month index calculated for the month: {month}. Value: {months.index(month)}')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return months.index(month)

    def encode_to_numeric(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        A function to change categorical variables to numerical value
        
        Parameters
        =--------=
        data: pandas data frame
            The main data frame
        columns: list
            The list of features to be label encoded
        
        Returns
        =-----=
        data: pandas dataframe
            The data frame with selected features label encoded
        """
        try:
            lb = preprocessing.LabelEncoder()
            for cols in columns:
                data[cols] = lb.fit_transform(data[cols])
                self.logger.info(f'feature: {cols} label encoded')
                print(f'feature: {cols} label encoded')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)
        finally:
            return data

    def save_data(self, df: pd.DataFrame, path: str, type: str = 'csv', index: bool = False) -> None:
        """
        A function to save data frames to file

        Parameters
        =--------=
        df: pandas data frame
            The data frame to save
        path: string
            The path of the file to save to
        type: string
            The type of the file to be saved as
        index: bool
            Whether the file to be saved will have an index or not

        Returns
        =-----=
        None: nothing
            Just saves the data frame to file
        """
        try:
            if type == 'csv':
                df.to_csv(path, index=index)
                self.logger.debug(f'data frame with shape: {df.shape} saved as a csv file at path: {path} successfully')
                print(f'data frame with shape: {df.shape} saved as a csv file at path: {path} successfully')
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)





    # TODO: NEW ADDITIONS
    # get state holiday list
    # 10 days for Easter
    # 3 days for public holiday
    # Considering christmas lasts for 12 days, Easter for 50 days and public holidays for 1 day.
    #a = public holiday, b = Easter holiday, c = Christmas, 0 = None
    def affect_list(self, change_list, interval, duration, index):
        start_pt = int(index-duration/2) - interval
        try:
            for index in range(start_pt, start_pt + interval):
                change_list[index] = 'before'
            for index in range(start_pt + interval, start_pt + interval + duration):
                change_list[index] = 'during'
            for index in range(start_pt + interval + duration, start_pt + interval + duration + interval):
                change_list[index] = 'after'
        except:
            pass
        return change_list

    def modify_holiday_list(self, holiday_list:list) -> list:
        new_index = ["neither"] * len(holiday_list)
        for index , value in enumerate(holiday_list):
            if value == 'a': #public holiday
                self.affect_list(new_index, 3, 1, index)
            elif value == 'b': #Easter
                self.affect_list(new_index, 10, 50, index)
            elif value == 'c': # christmas
                self.affect_list(new_index, 5, 12, index)
            else:
                pass
        return new_index

    def change_columns_type_to(self, df : pd.DataFrame, cols: list, data_type: str) -> pd.DataFrame:
        """
        Returns a DataFrame where the specified columns data types are changed to the specified data type
        Parameters
        ----------
        cols:
            Type: list
        data_type:
            Type: str
        Returns
        -------
        pd.DataFrame
        """
        try:
            for col in cols:
                df[col] = df[col].astype(data_type)
                self.logger.info(f"Successfully changed column: {col} type to {data_type}")
        except Exception as e:
            self.logger.error(e, trace_info=True)
            print(e, 'Failed to change columns type')
        finally:
            return df 

    def optimize_df(self, df : pd.DataFrame) -> pd.DataFrame:
        """
        Returns the DataFrames information after all column data types are optimized (to a lower data type)
        Parameters
        ----------
        None
        Returns
        -------
        pd.DataFrame
        """
        data_types = df.dtypes
        optimize = ['float64', 'int64']
        try:
            for col in data_types.index:
                if(data_types[col] in optimize):
                    if(data_types[col] == 'float64'):
                        # downcast a float column
                        df[col] = pd.to_numeric(
                            df[col], downcast='float')
                    elif(data_types[col] == 'int64'):
                        # downcast an integer column
                        df[col] = pd.to_numeric(
                            df[col], downcast='unsigned')
            self.logger.info(f"DataFrame optimized")
            print(f"DataFrame optimized")
        except Exception as e:
            self.logger.error(e, trace_info=True)
            print(e, 'Failed to optimize')
        finally:
            return df