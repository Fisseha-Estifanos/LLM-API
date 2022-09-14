"""
A script to visualize data.
"""

# imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging


class dataVisualizer():
    """
    A data visualizer class.
    """
    def __init__(self, fromThe: str) -> None:
        """
        The data visualizer initializer

        Parameters
        =--------=
        fromThe: string
            The file importing the data visualizer

        Returns
        =-----=
        None: nothing
            This will return nothing, it just sets up the data visualizer
            script.
        """
        try:
            # setting up logger
            self.logger = self.setup_logger('../logs/visualizer_root.log')
            self.logger.info('\n    #####-->    Data visualizer logger for ' +
                             f'{fromThe}    <--#####\n')
            print('Data visualizer in action')
        except Exception as e:
            print(e)

        # setting up seaborn styles
        # pals = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind']
        # sns.color_palette(palette='pastel')
        # TODO: remove any other color
        sns.set_theme(style="darkgrid")
        # TODO: add try catch to all visualizer functions
        # TODO: add comments to all visualizer functions
        # TODO: modify all log messages properly
        # TODO: add save_as parameter just like the plot_count function for
        # all functions
        # TODO: PEP8

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

    # TODO : add the name and things from last weeks pie plot function
    def plot_pie(self, df: pd.DataFrame, column: str, title: str = '',
                 largest: int = 10, save_as: str = '') -> None:
        """
        A function to plot pie charts

        Parameters
        =--------=

        Returns
        =-----=
        None: nothing
            Only plots the plot
        """
        # TODO : resolve this fig variable it is not being used
        # fig = plt.figure(figsize=(10, 10))
        col = df[column].value_counts().nlargest(n=largest)

        data = col.values
        labels = col.keys()

        last_num = len(data)

        colors = sns.color_palette('muted')[0:last_num]

        plt.pie(data, labels=labels, colors=colors, autopct='%.000f%%')
        if title == '':
            plt.title(f'{column} pie plot')
            self.logger.info(f'{column} pie plot plotted successfully')
        else:
            plt.title(title)
            self.logger.info(f'{title} pie plot plotted successfully')
        if save_as == '':
            plt.show()
        else:
            plt.savefig(save_as)

    def plot_hist(self, df: pd.DataFrame, column: str, color: str) -> None:
        # plt.figure(figsize=(15, 10))
        # fig, ax = plt.subplots(1, figsize=(12, 7))
        self.logger.info('setting up distribution plot')
        sns.displot(data=df, x=column, color=color, kde=True, height=7,
                    aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()
        # TODO: if logger info is bad try this
        # logger.info(f'Distribution of {column} plot successfully plotted')
        self.logger.info(f'{column} hist plot plotted successfully')

    def plot_count(self, df: pd.DataFrame, column: str, hue: str = '',
                   title: str = '', save_as: str = '') -> None:
        self.logger.info('setting up count plot')
        plt.figure(figsize=(12, 7))
        if hue == '':
            sns.countplot(data=df, x=column)
        else:
            sns.countplot(data=df, x=column, hue=hue)
        if title == '':
            plt.title(f'Distribution of {column}', size=20, fontweight='bold')
            self.logger.info(f'{column} count plot plotted successfully')
        else:
            plt.title(f'Distribution of {title}', size=20, fontweight='bold')
            self.logger.info(f'{title} count plot plotted successfully')
        plt.xlabel(f'{column}', fontsize=16)
        plt.ylabel("Count", fontsize=16)
        plt.xticks(rotation=45)
        if save_as == '':
            plt.show()
        else:
            plt.savefig(save_as)
        # TODO: if logger info is bad try this
        # logger.info(f'Distribution of {column} plot successfully plotted')

    def plot_bar(self, df: pd.DataFrame, x_col: str, y_col: str, title: str,
                 xlabel: str, ylabel: str) -> None:
        self.logger.info('setting up bar plot')
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.show()
        self.logger.info(f'{title} bar plot plotted successfully')

    # TODO : update this correlation map with the one from last week and
    # compare it with the new one below
    def plot_heatmap(self, df: pd.DataFrame, title: str, cbar: bool = False,
                     save_as: str = '') -> None:
        self.logger.info('setting up heat map plot')
        plt.figure(figsize=(12, 7))
        sns.heatmap(df.corr(), annot=True, fmt='.5f', linewidths=1, cbar=True)
        plt.title(title, size=20, fontweight='bold')
        if save_as == '':
            plt.show()
        else:
            plt.savefig(save_as)
        self.logger.info(f'{title} heat map plot plotted successfully')

    def plot_heatmap_from_correlation(self, correlation, title: str):
        '''
        heatmap: Plot rectangular data as a color-encoded matrix and correlation matrix.
        title: Title of the plot
        correlation: correlation matrix
        '''
        plt.figure(figsize=(14, 9))
        sns.heatmap(correlation)
        plt.title(title, size=18, fontweight='bold')
        plt.show()

    def plot_box(self, df: pd.DataFrame, x_col: str, title: str) -> None:
        self.logger.info('setting up box plot')
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.show()
        self.logger.info(f'{title} box plot plotted successfully')

    def plot_box_multi(self, df: pd.DataFrame, x_col: str, y_col: str,
                       title: str) -> None:
        self.logger.info('setting up box plot')
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        self.logger.info(f'{title} multi box plot plotted successfully')

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str,
                     title: str, hue: str, style: str) -> None:
        """
        # scatter: Plot data as a scatter plot.
        # df: dataframe to be plotted
        # x_col: x-axis column
        # y_col: y-axis column
        # title: Title of the plot
        # hue: hue column
        """
        self.logger.info('setting up scatter plot')
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        self.logger.info(f'{title} scatter plot plotted successfully')

    def bi_variate_plot(self, df, features, fields):
        fig, axs = plt.subplots(10, 3, figsize=(20, 45))
        for col in range(len(features)):
            for f in range(len(fields)):
                self.logger.info('setting up hist plot')
                sns.histplot(df, x=features[col]+"_"+fields[f],
                             hue="diagnosis", element="bars", stat="count",
                             # TODO : modify palette
                             palette=["gold", "purple"],
                             ax=axs[col][f])
        self.logger.info('several bi-variate plots plotted successfully')





    # TODO: NEW ADDITIONS
    def plotly_plot_pie(self, df, column, limit=None, title=None):
        a = pd.DataFrame({'count': df.groupby([column]).size()}).reset_index()
        a = a.sort_values("count", ascending=False)
        if limit:
            a.loc[a['count'] < limit, column] = f'Other {column}s'
        if title == None:
            title=f'Distribution of {column}'
        fig = px.pie(a, values='count', names=column, title=title, width=800, height=500)
        fig.show()

    def plot_factor(self, data: pd.DataFrame, x: str, y: str, col: str,
                    palette: str, hue: str, col_order: list, 
                    title:str) -> None:
        """
        """
        try:
            self.logger.info('setting up factor plot')
            #plt.figure(figsize=(12, 7))
            sns.factorplot(data= data, x=x, y=y, col=col, palette=palette, 
                        hue=hue, col_order=col_order, title=title)
            plt.show()
        except Exception as e:
            self.logger.error(e, exec_info=True)
            print(e)

    def plotly_plot_hist(self, df, column, color=['cornflowerblue'], title=None):
        if title == None:
            title=f'Distribution of {column}'
        fig = px.histogram(
                df,
                x=column,
                marginal='box',
                color_discrete_sequence=color,
                title=title)
        fig.update_layout(bargap=0.01)
        fig.show()