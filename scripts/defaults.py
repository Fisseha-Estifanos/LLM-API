"""
A default script for all default variables
"""

# base repository
repo = '../'

# the path to the data folder
data_path = 'data/'

# setting up dvc data and folder structure paths
# main data file name
news_data_file = 'example_data.csv'
job_train_data_file = 'relations_dev.json'
job_test_data_file = 'relations_test.json'

# the local data path
news_local_path = repo + data_path + news_data_file
job_train_local_path = repo + data_path + job_train_data_file
job_test_local_path = repo + data_path + job_test_data_file

# the path to the data set
store_path = data_path + news_data_file
train_path = data_path + job_train_data_file
test_path = data_path + job_test_data_file

# the path to the plots folder
plot_path = 'plots/'
