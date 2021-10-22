import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


def load_dataset(url):
    data = pd.read_csv(url)
    data.set_index("Time", inplace=True)
    data.index = pd.to_datetime(data.index)
    data.sort_index(axis=0, inplace=True)
    return data


def lagged_df(series, num_of_lags, keep_first=False):
    df = pd.DataFrame(series)
    df.columns = ["lag_0"]
    for i in range(1, num_of_lags + 1):
        df["lag_{}".format(i)] = df["lag_0"].shift(i)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    if keep_first == False:
        df = df.iloc[:, 1:]

    return df


def returns(data):
    lagged = lagged_df(data, 1, keep_first=True)
    X = (lagged["lag_0"] - lagged["lag_1"]) / lagged["lag_1"]  # stationizing the data
    X.index = lagged.iloc[:, 0]
    return X


def bin_time(t, window=4):
    for hour in range(0, 21, window):
        start = hour
        end = hour + 3
        if t.hour >= start and t.hour <= end:
            return "{}-{}".format(start, end + 1)


def visualize(vals, xlim=(-0.05, 0.05), kind="kdeplot"):
    day_returns1 = pd.DataFrame(vals)
    day_returns1.index = vals.index.weekday  # changing the timestamp to a weekday code
    day_returns1.columns = ["lag_0"]  # renaming the column

    day_of_week = {0: "Mon", 1: "Tues", 2: "Wed", 3: "Thur", 4: "Fri", 5: "Sat", 6: "Sun"}
    sns.set(rc={'figure.figsize': (9, 4)})

    day_returns1 = day_returns1.sort_index()
    days = day_returns1.groupby(day_returns1.index)["lag_0"]

    if kind == "kdeplot":

        for i in range(7):
            g = sns.kdeplot(day_returns1["lag_0"][day_returns1.index == i], label=day_of_week[i])
            g.set(xlim=xlim)
        plt.figure(figsize=(100, 200))
        plt.show()

    elif kind == "bar":

        # days.std()
        days.mean().plot(kind='bar', figsize=(8, 8),
                         title='Days of Week Average Returns')
        plt.show()

    elif kind == "error":
        # day_returns1 = day_returns1.sort_index()
        days = day_returns1.groupby(day_returns1.index)["lag_0"]
        # days.std()
        days.mean().plot(kind='bar', yerr=days.std(), figsize=(8, 8),
                         title='Days of Week Average Returns')
        plt.show()

    elif kind == "mean":
        return pd.DataFrame(days.mean()).style.background_gradient(cmap='RdYlGn')

    elif kind == "time_bar":
        day_returns1 = pd.DataFrame(vals)
        day_returns1.columns = ["lag_0"]  # renaming the column
        cp = day_returns1.copy()
        cp.index = cp.index.to_series().apply(bin_time)

        windows = ["{}-{}".format(hour, hour + 4) for hour in range(0, 21, 4)]
        mapping = {window: i for i, window in enumerate(windows)}
        cp = cp.sort_index()
        cp["mapping"] = cp.index.map(mapping)
        cp.sort_values(by="mapping", inplace=True)

        groups = cp.groupby(cp.index, sort=False)["lag_0"]
        groups.mean().plot(kind='bar', figsize=(8, 8),
                           title='Hours of Day Average Returns')
        # groups.mean().plot(kind ='bar' , yerr = groups.std(), figsize = (8,8),
        #            title =  'Hours of day Average Returns')
        plt.show()

    elif kind == "time_mean":
        day_returns1 = pd.DataFrame(vals)
        day_returns1.columns = ["lag_0"]  # renaming the column
        cp = day_returns1.copy()
        cp.index = cp.index.to_series().apply(bin_time)

        windows = ["{}-{}".format(hour, hour + 4) for hour in range(0, 21, 4)]
        mapping = {window: i for i, window in enumerate(windows)}
        cp = cp.sort_index()
        cp["mapping"] = cp.index.map(mapping)
        cp.sort_values(by="mapping", inplace=True)

        groups = cp.groupby(cp.index, sort=False)["lag_0"]
        return pd.DataFrame(groups.mean()).style.background_gradient(cmap='RdYlGn')


def make_datasets(data, num_lags=5, lagged_col="Last", cols=None, transform=None):
    if transform == "log":
        X = lagged_df(np.log(data[lagged_col]), num_lags, keep_first=False)
    else:
        X = lagged_df(data[lagged_col], num_lags, keep_first=False)

    if cols is not None:
        for col in cols:
            X[col] = data[col].values[num_lags:]
            X = X[X[col] > 0]
            if col == "Volume":
                X["Volume"] = np.log(X["Volume"])

    X.reset_index(inplace=True, drop=True)
    y = X.pop("lag_0")
    y = y.values
    return X.values, y


def makeXY(data, num_lags=5, lagged_col="Last", returns=True, transform=None):
    data_to_lag = data[lagged_col].values
    if returns == True:
        data_to_lag = lagged_df(data_to_lag, 1, keep_first=True)
        data_to_lag = (lagged["lag_0"] - lagged["lag_1"])
    if transform == "log":
        transformer = lambda x: np.log(x)
    elif transform == "arcsinh":
        transformer = lambda x: np.arcsinh(x)
    else:
        transformer = lambda x: x

    X = lagged_df(transformer(data_to_lag), num_lags, keep_first=False).values
    vol = lagged_df(data['Volume'], 1, keep_first=False).values[:, 1]
    vol = vol[num_lags - 1:]
    X = X[vol > 0]
    X = np.hstack((X, np.log(vol[vol > 0]).reshape(-1, 1)))
    # X = np.hstack((X, vol[vol > 0].reshape(-1, 1)))
    # y = data['Change'].values[num_lags:].reshape(-1, 1)
    # y = y[vol > 0]
    y = X[:, 0]
    X = X[:, 1:].copy()  # remove lag 0
    # y = np.where(y > 0,1,0)

    return X, y


def make_log_returns_series(data, num_lags=10, lagged_col="Close", truncate=0.03, x_as_series=False):
    X = data[lagged_col].values
    vol = data['Volume'].values

    X = X[vol > 0]
    X = lagged_df(X, 1, keep_first=True)
    X = np.log((X["lag_0"] / X["lag_1"]).values)
    X = np.tanh(X / truncate) * truncate  # truncate to +- truncate
    X = lagged_df(X, num_lags, keep_first=False).values
    y = X[:, 0].copy()
    X = X[:, 1:].copy()  # remove lag 0

    vol = vol[vol > 0]
    vol = lagged_df(vol, 1, keep_first=False)
    vol = np.log((vol["lag_0"] / vol["lag_1"]).values)
    vol = np.tanh(vol) * truncate
    vol = vol[num_lags:].reshape(-1, 1)
    X = np.hstack((X, vol))
    return X, y.ravel()


"""    if x_as_series:
      X = data[lagged_col].values
      vol = data['Volume'].values
      X = X[vol > 0]
      X = lagged_df(X, 1, keep_first=True).values[:,1]
      vol = vol[vol > 0]
      vol = lagged_df(vol, 1, keep_first=False).values[:, 1]
      vol = vol[num_lags:]
      X = np.log(X)
      X = lagged_df(X, num_lags, keep_first=False).values
      X = X[:,1:].copy() # remove lag 0
      X = np.hstack((X, np.log(vol).reshape(-1, 1)))

    return X, y.ravel()"""


def profitibility(y_true, y_pred, proportional=1, bet_scaler=100, bound=0):
    signs = np.where(y_pred > 0, 1, -1)
    if proportional == 0 and bound == 0:
        bets = signs
    elif proportional == 0 and bound > 0:
        bets = np.where(y_pred >= bound, 1, y_pred)
        bets = np.where(bets <= -1 * bound, -1, bets)
        bets = np.where(abs(bets) == 1, bets, 0)
    else:
        bets = y_pred * bet_scaler
    return np.cumsum(y_true * bets), bets

def reshape(data):
    return np.reshape(data, (data.shape[0], 1, data.shape[1]))


class GroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_size : int, default=None
        Maximum size for a single training set.

    --------
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        group_test_size = n_groups // n_folds
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end -
                                          self.max_train_size:train_end]
            for test_group_idx in unique_groups[group_test_start:
            group_test_start +
            group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)
            yield [int(i) for i in train_array], [int(i) for i in test_array]


class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):

        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        # if groups is None:
        #    raise ValueError(
        #        "The 'groups' parameter should not be None")
        groups = range(len(X))
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                    np.concatenate((train_array,
                                    train_array_tmp)),
                    axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
            group_test_start +
            group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                    np.concatenate((test_array,
                                    test_array_tmp)),
                    axis=None), axis=None)

            test_array = test_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]


def stacked_dataset(X, y, models, grids, test_size=(48 * 7), scaler=MinMaxScaler(), scale_y=True, include_volume=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_scaler = scaler.fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    if scale_y == True:

        if type(y_train) == type(X):
            y_scaler = MinMaxScaler().fit(y_train.values.reshape(-1, 1))
            y_train = y_scaler.transform(y_train.values.reshape(-1, 1)).ravel()
            y_test = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
        elif type(y_train) == type(np.array([0])):
            y_scaler = scaler.fit(y_train.reshape(-1, 1))
            y_train = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_train = y_train.values
        y_test = y_test.values

    kf = KFold(n_splits=4, shuffle=False)
    X_train_copy = pd.DataFrame(X_train)
    X_test_copy = pd.DataFrame(X_test)

    if include_volume == True:
        X_train_copy = pd.DataFrame(X_train_copy.iloc[:, [0, -1]])
        X_test_copy = pd.DataFrame(X_test_copy.iloc[:, [0, -1]])
    else:
        X_train_copy = pd.DataFrame(X_train_copy.iloc[:, [0]])
        X_test_copy = pd.DataFrame(X_test_copy.iloc[:, [0]])

    for model in models:

        if model == "PT_MLP":
            n_jobs = 1
        else:
            n_jobs = -1

        print("*** Training {} ***".format(model))
        train_pred_st = []
        test_pred_st = []
        count = 1
        for split in kf.split(X_train):  # should be X_train
            print("[{}/4 splits done for {}]".format(count, model))
            train_index = list(split[0])
            test_index = list(split[1])

            X_train_st = X_train[train_index, :]
            X_test_st = X_train[test_index, :]
            y_train_st = y_train[train_index]  # .values.reshape(-1,1)

            model_gs = GridSearchCV(models[model], grids[model], cv=kf, verbose=0, n_jobs=n_jobs)
            model_gs.fit(X_train_st, y_train_st)
            preds = model_gs.predict(X_test_st)
            # preds = scaler.inverse_transform(preds.reshape(-1, 1)).ravel() # This reverts the predictions back to the unscaled values
            train_pred_st += preds.tolist()
            count += 1

        print()
        print("Done Fitting splits, Predicting Test Data")
        print()
        model_gs_all = GridSearchCV(models[model], grids[model], cv=kf, verbose=0, n_jobs=n_jobs)
        model_gs_all.fit(X_train, y_train)
        test_pred_st = model_gs_all.predict(X_test)

        # test_pred_st = scaler.inverse_transform(test_pred_st.reshape(-1,1)).ravel()
        # train_pred_st = np.array(train_pred_st).reshape(-1,1)

        # scaler = scaler.fit(train_pred_st)
        # train_pred_st = scaler.transform(train_pred_st).ravel()
        # test_pred_st = scaler.transform(test_pred_st.reshape(-1,1)).ravel()

        # X_train_copy.loc[:, model] = X_scaler.transform(np.array(train_pred_st).reshape(-1,1)).ravel()
        X_train_copy.loc[:, model] = train_pred_st
        X_test_copy.loc[:, model] = test_pred_st
        print("Finished {}".format(model))

    print("Done Generating")

    y_train = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    X_train_copy = X_train_copy.astype('float32')
    X_test_copy = X_test_copy.astype('float32')

    return X_train_copy, X_test_copy, y_train, y_test


def create_level0_dataset(X, y, n_splits=10, num_lags=5, models=None, grids=None, discard_size=0.15, test_size=48 * 7,
                          scaler=MinMaxScaler(), pca_inputs=False, pca_outputs=False):
    count = 1

    X_discard, X_validtest, y_discard, y_validtest = train_test_split(X, y, train_size=discard_size, shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X_validtest, y_validtest, test_size=test_size, shuffle=False)

    y_discard = y_discard.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)

    splits_copy = PurgedGroupTimeSeriesSplit(n_splits=n_splits, group_gap=0).split(X_train)
    cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, group_gap=num_lags)

    start_index = next(iter(splits_copy))[1][0]  # Cut off point for discarding train data
    stacked_X_train = X_train[start_index:, :].copy()
    stacked_X_test = X_test[:, :].copy()  # lag1 and volume
    stacked_y_train = y_train[start_index:].copy()

    for mod in models:
        print(" ***** Training {} ***** ".format(mod))
        train_preds = []
        counter = 0
        splits = PurgedGroupTimeSeriesSplit(n_splits=n_splits, group_gap=0).split(X_train)
        for train_idx, test_idx in splits:

            X_train_split = np.vstack((X_discard, X_train[train_idx, :]))
            X_test_split = X_train[test_idx, :]

            y_train_split = np.vstack((y_discard, y_train[train_idx])).ravel()
            y_test_split = y_train[test_idx].ravel()

            model = models[mod]
            pipe = Pipeline(steps=[
                ('scaler', scaler),
                ('model', model)
            ])

            search = GridSearchCV(pipe, grids[mod], cv=cv, verbose=0, n_jobs=-1)
            search.fit(X_train_split, y_train_split.ravel())
            print(search.best_params_)
            train_preds += search.predict(X_test_split).tolist()
            counter += 1
            print(" ***** Done [{}/{}] splits for {} ***** \n".format(counter, n_splits, mod))

        print("Done Fitting Splits, Predicting Test Data for {}".format(mod))
        train_preds = np.array(train_preds).reshape(-1, 1)
        stacked_X_train = np.hstack((stacked_X_train, train_preds))
        # print("stacked_X_train shape: {}".format(stacked_X_train.shape))
        # print("train preds shape: {}".format(np.array(train_preds).shape))

        search_all = GridSearchCV(pipe, grids[mod], cv=cv, verbose=0, n_jobs=-1)

        X_search = np.vstack((X_discard, X_train))
        y_search = np.vstack((y_discard.reshape(-1, 1), y_train.reshape(-1, 1)))
        search_all.fit(X_search, y_search.ravel())
        test_preds = search_all.predict(X_test).reshape(-1, 1)
        stacked_X_test = np.hstack((stacked_X_test, test_preds))

    print("\nFinished Generating Dataset\n")

    return stacked_X_train, stacked_X_test, stacked_y_train.ravel(), y_test.ravel()

def plot_strats(y_true, y_pred, name = "In sample"):
    plt.plot(np.cumsum(y_pred * y_true), label="{} Proportional".format(name))
    plt.plot(np.cumsum(np.where(y_pred > 0, 1, -1) * y_true), label="{} LongShort".format(name))
    plt.plot(np.cumsum(y_true), label="{} True".format(name))
    plt.legend()

def merton_loss(y_true, y_pred):
    return np.mean((y_pred * y_true - 1) ** 2)

def print_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    merton = merton_loss(y_true, y_pred)
    print('RMSE: {0:.7f}'.format(rmse))
    print('MAE: {0:.7f}'.format(mae))
    try:
        print('MAPE: {0:.3f}'.format(mape))
    except:
        pass
    print('r2: {0:.7f}'.format(r2))
    print('Merton: {0:.7f}'.format(merton))
    print(classification_report(np.where(y_true > 0, 1, 0), np.where(y_pred > 0, 1, 0)))
    print(confusion_matrix(np.where(y_true > 0, 1, 0), np.where(y_pred > 0, 1, 0)))

def plot_pie(vals):
    labels = ["Up", "Down"]
    positives = sum(np.where(vals > 0, 1, 0)) / len(vals)
    sizes = [positives, 1 - positives]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def split_sequences(X, y, n_steps):
    sequences = np.hstack((X, y.reshape(-1, 1)))
    X = []
    y = []
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
