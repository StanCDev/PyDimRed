"""
This module provides various functions to visualize and evaluate the performance of
dimensionality reduction (DR) models using different plots. The functions included
can display relational plots, scatter plots, heatmaps, line plots, and bar plots to help in analyzing the performance
of models such as TSNE, TRIMAP, and others.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from .exceptions import checkCondition, checkDimensionCondition


def display(
    X: np.array,
    y: np.array,
    marker_size=5,
    title: str = None,
    x_label: str = "x",
    y_label: str = "y",
    hue_label: str = "z",
    figsize: tuple = None,
) -> None:
    """
    Display 3 dimensional data on a seaborn.relplot.
    Two input feature dimensions in X and one output dimension in Y

    Args:
    -----
        X (np.array): N x 2 dimensional array of feature data

        y (np.array): N dimensional array of label / output data

        marker_size (int) : size of markers on seaborn.relplot, default=5

        title (str) : plot title. default=None

        x_label (str): x axis label name

        y_label (str): y axis label name

        hue_label (str): name of color hue variable

        figsize (tuple): x and y figure size in inches. Default = None

    Returns:
    --------
        None
    """
    checkDimensionCondition(
        X.shape[0] == y.shape[0],
        "Train data doesn't have the same number of samples (dimension 0 doesn't correspond)",
        X.shape,
        y.shape,
    )
    checkDimensionCondition(
        X.shape[1] == 2,
        "X is not a 2 dimensional dataset (dimension 1 doesn't correspond)",
        X.shape[1],
        2,
    )
    checkDimensionCondition(
        len(y.shape) == 1,
        "y is not a 1 dimensional dataset (dimension 1 doesn't correspond)",
        y.shape,
        1,
    )
    checkCondition(
        marker_size > 0,
        f"Marker size = {marker_size}, cannot be smaller than or equal to 0",
    )

    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize

    df_X = pd.DataFrame(X, columns=[x_label, y_label])
    df_Y = pd.DataFrame(y, columns=[hue_label])
    df = pd.concat([df_X, df_Y], axis=1)

    sns.set_theme()
    plot = sns.relplot(
        data=df, x=x_label, y=y_label, hue=hue_label, s=marker_size, linewidth=0
    )
    if title is not None:
        plot.set(title=title)
    plt.show()
    return


def display_group(
    names: list[str],
    X_train_list: list[np.array],
    y_train: np.array,
    X_test_list: list[np.array] = None,
    y_test: np.array = None,
    nbr_cols: int = 3,
    nbr_rows: int = 4,
    marker_size=5,
    legend: str = "full",  # change discrete (categorical) / cts
    title: str = None,
    x_label: str = "x",
    y_label: str = "y",
    hue_label: str = "z",
    grid_x_label: list[str] = None,
    grid_y_label: list[str] = None,
    figsize: tuple = None,
) -> None:
    """
    Given a list of train data (optionally test data) with the same labels (must all be in same order) create a multi scatter plot of all
    data on a grid. If both XtestList and ytest are not None then train data and test data will be differentiated via
    different markers. Another option that can be specified is the use of a 'global' grid on the graph. If grid_x_label or grid_y_label are
    a list of names grid like naming of each suplot will occur

    Args:
    -----
        names (list[str]): Title of each subplot. Subplots don't have titles when default = None

        X_train_list (list[np.array]): List of N x 2 dimensional data sets to be plotted

        ytrain (np.array): N dimensional array of label / output data

        X_test_list (list[np.array]): Optional list of N x 2 dimensional test data sets to be plotted

        ytest (np.array): Optional N dimensional array of label / output data

        nbr_cols (int): Number of columns in the grouped plot, will have at most nbrCols graphs stacked vertically

        nbr_rows (int): Number of rows in the grouped plot, will have at most nbrRows graphs stacked horizontally

        marker_size (int) : size of markers on seaborn.relplot, default=5

        legend (str): seaborn legend argument. "full" show each different value on legend, "auto" will make seaborn decide

        x_label (str): x axis label name

        y_label (str): y axis label name

        hue_label (str): name of color hue variable

        grid_x_label (list[str]) : x-axis labels for global grid of subplots. default = None,

        grid_y_label (list[str]) : y-axis labels for global grid of subplots. default = None

        figsize (tuple): x and y figure size in inches. Default = None

    Returns:
    --------
        None
    """
    if names is not None:
        checkCondition(
            len(X_train_list) == len(names),
            f"The number of method names = {
                len(X_train_list)} does not match the number of reduced data sets = {
                len(names)}",
        )
    if X_test_list is not None and y_test is not None:
        checkCondition(
            len(X_test_list) == len(X_train_list),
            f"Number of train datasets = {
                len(X_test_list)} must match number of test datasets = {
                len(X_train_list)}",
        )
    checkCondition(
        len(X_train_list) <= nbr_cols * nbr_rows,
        f"Not enough plot boxes to plot all datasets. {
            len(X_train_list)} datasets but \
        {nbr_cols} columns and {nbr_rows} rows = {
            nbr_cols * nbr_rows} is max number of datasets",
    )
    checkCondition(
        marker_size > 0,
        f"marker_size = {marker_size}, cannot be smaller than or equal to 0",
    )
    checkCondition(
        nbr_cols > 0, f"nbr_cols = {nbr_cols}, cannot be smaller than or equal to 0"
    )
    checkCondition(
        nbr_rows > 0, f"nbr_rows = {nbr_rows}, cannot be smaller than or equal to 0"
    )
    if grid_x_label is not None:
        checkCondition(
            len(grid_x_label) == nbr_cols,
            f"Length of grid_x_label must correspond to the number of columns,\
            as there is one x label per column. nbr_cols = {nbr_cols}, number of grid_x_labels = {len(grid_x_label)}",
        )
    if grid_y_label is not None:
        checkCondition(
            len(grid_y_label) == nbr_rows,
            f"Length of grid_y_label must correspond to the number of rows,\
            as there is one y label per row. nbrRows = {nbr_rows}, number of grid_y_label = {len(grid_y_label)}",
        )
    N = y_train.shape[0]  # number of train samples

    N_test = 0  # number of test samples
    if X_test_list is not None and y_test is not None:
        N_test = y_test.shape[0]

    # sharex=True,sharey=True
    fig, axs = plt.subplots(ncols=nbr_cols, nrows=nbr_rows)

    if title is not None:
        fig.suptitle(title)

    sns.set_theme()
    plt.rcParams["savefig.dpi"] = 300
    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize

    for i in range(nbr_rows):
        for j in range(nbr_cols):
            ax = axs[i, j]
            if i == 0 and grid_x_label is not None:
                ax.set_xlabel(grid_x_label[j])
                ax.xaxis.set_label_position("top")

            if j == 0 and grid_x_label is not None:
                ax.set_ylabel(grid_y_label[i])

            ax.set_frame_on(b=False)
            ax.set_xticks([])
            ax.set_yticks([])
            # axs[i,j].grid(visible=False)
            # axs[i, j].axis('off')

    if X_test_list is not None and y_test is not None:
        Y = np.concatenate((y_train, y_test), axis=0)

    for i, X_train in enumerate(X_train_list):
        checkDimensionCondition(
            X_train.shape[0] == y_train.shape[0],
            "Train data doesn't have the same number of samples (dimension 0 doesn't correspond)",
            X_train.shape,
            y_train.shape,
        )
        ax = axs[i // nbr_cols, i % nbr_cols]

        df_X = None
        df_Y = None

        if X_test_list is not None and y_test is not None:
            checkDimensionCondition(
                X_test_list[i].shape[0] == y_test.shape[0],
                "Train data doesn't have the same number of samples (dimension 0 doesn't correspond)",
                X_test_list[i].shape,
                y_test.shape,
            )

            N_train, N_test = X_train.shape[0], X_test_list[i].shape[0]

            # Array of ones and zeros that will be concatenated to determine if
            # data point is train or test. 1 = train, 0 = test
            train = np.full(N_train, 0).reshape(-1, 1)

            test = np.full(N_test, 1).reshape(-1, 1)
            # print(f"Train and test size: {train.shape} , {test.shape}")
            train_test_label = np.concatenate(
                (train, test), axis=0)  # .reshape(-1,1)
            # print(f"Concat size : {trainTestLabel.shape}")

            X = np.concatenate((X_train_list[i], X_test_list[i]), axis=0)
            X = np.concatenate((X, train_test_label), axis=1)

            df_X = pd.DataFrame(
                X,
                columns=[
                    x_label,
                    y_label,
                    "train (1) or test (0)"])
            df_Y = pd.DataFrame(Y, columns=[hue_label])

        else:
            df_X = pd.DataFrame(X_train_list[i], columns=[x_label, y_label])
            df_Y = pd.DataFrame(y_train, columns=[hue_label])

        df = pd.concat([df_X, df_Y], axis=1)

        plot = None
        if X_test_list is not None and y_test is not None:
            plot = sns.scatterplot(
                data=df,
                x=x_label,
                y=y_label,
                hue=hue_label,
                s=marker_size,
                style="train (1) or test (0)",
                linewidth=0,
                ax=ax,
                legend=legend,
            )
        else:
            plot = sns.scatterplot(
                data=df,
                x=x_label,
                y=y_label,
                hue=hue_label,
                s=marker_size,
                linewidth=0,
                ax=ax,
                legend=legend,
            )

        ax.get_legend().remove()
        if names is not None:
            plot.set(title=names[i])

    ax = axs[0, 0]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")

    plt.grid(False)
    plt.show()
    return


def display_heatmap(
    X: np.array,
    x_range: list,
    y_range: list,
    x_label: str = "x",
    y_label: str = "y",
    title: str = None,
    figsize: tuple = None,
) -> None:
    """
    Plot values of a 2D array on a heatmap with range of x and y values and corresponding feature names

    Args:
    -----
        X (np.array) : N1 x N2 two dimensional array of values.

        x_range (list) : list of values for feature x of length N1. Each value corresponds to a row

        y_range (list) : list of values for feature y of length N2. Each value corresponds to a column

        x_label (str): x axis label name

        y_label (str): y axis label name

        figsize (tuple): x and y figure size in inches. Default = None

    Returns:
    --------
        None
    """
    checkDimensionCondition(
        len(X.shape) == 2,
        "X is not a 2 dimensional dataset (dimension 1 doesn't correspond)",
        X.shape,
        "(x1,x2)",
    )
    checkDimensionCondition(
        X.shape[0] == len(x_range),
        "Number of y features doesn't correspond to number of y values",
        X.shape[0],
        len(y_range),
    )
    checkDimensionCondition(
        X.shape[1] == len(y_range),
        "Number of x features doesn't correspond to number of x values",
        X.shape[1],
        len(x_range),
    )

    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize

    df = pd.DataFrame(X, columns=y_range, index=x_range)
    plot = sns.heatmap(df)

    plot.set_xlabel(x_label)
    # plt.xticks(rotation=30)
    plot.set_ylabel(y_label)
    # plt.xticks(rotation=30)

    if title is not None:
        plot.set_title(title)

    plt.show()

    return


def display_heatmap_df(
    df: pd.DataFrame,
    feature1: str,
    feature2: str,
    values: str,
    title: str = None,
    figsize: tuple = None,
) -> None:
    """
    Plot values of a data frame on a heatmap given feature column names and output column name

    Args:
    -----
        df (pd.df) : N1 x N2 two dimensional array of values.

        xRange (list) : list of values for feature x of length N1

        yRange (list) : list of values for feature y of length N2

        feature1 (str) : name of first feature (x)

        feature2 (str): name of second feature (y)

        values (str): name of values in data frame

        figsize (tuple): x and y figure size in inches. Default = None

    Returns:
    --------
        None
    """
    checkCondition(df is not None, "pandas Dataframe is None")

    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize

    df_matrix = df.pivot(index=feature1, columns=feature2, values=values)
    plot = sns.heatmap(df_matrix)
    if title is not None:
        plot.set_title(title)
    plt.show()

    return


def display_accuracies(
    names: list[str],
    accuracies: list[float],
    title: str = None,
    hue_label: str = "accuracy",
    figsize: tuple = None,
) -> None:
    """
    Simple function to plot accuracies and method name on a bar graph with color scale proportional to
    accuracy

    Args:
    -----
        names (list[str]) : name of each method

        accuracies (list[float]): accuracy for each method

        hue_label (str): name of color hue variable

        figsize (tuple): x and y figure size in inches. Default = None

    Return:
    -------
        None
    """
    checkCondition(
        len(names) == len(accuracies),
        f"Lengths of data are not equal: {
            len(names)} names while {
            len(accuracies)} accuracy values",
    )

    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize

    model_perf = pd.DataFrame({"method": names, "accuracy": accuracies})

    sns.set_theme()
    plt.rcParams["savefig.dpi"] = 300
    min_acc = np.min(accuracies)
    y_range = max(min_acc - 5, 0), 100
    bargraph = sns.barplot(
        data=model_perf,
        x="method",
        y="accuracy",
        hue=hue_label)
    bargraph.set_ylim(y_range)

    if title is not None:
        bargraph.set_title(title)

    plt.show()
    return


def display_training_validation(
    x: list,
    y: pd.DataFrame,
    x_name: str = "NumberNeighbours",
    title: str = None,
    figsize: tuple = None,
) -> None:
    """
    Line plot of accuracy vs. a common variable parameter for multiple methods

    Args:
    -----
        x (list): list of values x feature takes

        y (pd.DataFrame): Each column is a method, and rows are corresponding accuracies for that method at
        given parameter value

        x_name (str): Name of x feature being varied

        title (str): plot title. Default is None, no title

        figsize (tuple): x and y figure size in inches. Default = None

    Return:
    -------
        None

    Example:
    --------
    >>> parameters = [5, 10, 15, 20] # list of n_nbrs
    >>> y = {
    >>>    "TSNE" : [94.0990, 93.4519, 91.9385, 92.4469],
    >>>    "TRIMAP" : [82.5421, 82.1633, 82.5700, 82.3850]
    >>> }
    >>> y = pd.DataFrame(y)
    >>> displayTrainVal(parameters, y)
    """
    checkDimensionCondition(
        len(x) == y.shape[0],
        "DataFrames must have the same number of rows",
        len(x),
        y.shape[0],
    )

    if figsize is not None:
        plt.rcParams["figure.figsize"] = figsize

    x_df = pd.DataFrame({x_name: x})
    merged_df = pd.concat([x_df, y], axis=1)
    df = pd.melt(merged_df, [x_name], var_name="Method", value_name="Accuracy")

    sns.set_theme()
    plot = sns.lineplot(data=df, x=x_name, y="Accuracy", hue="Method")

    if title is not None:
        plot.set_title(title)

    plt.show()
    return
