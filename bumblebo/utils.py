import sklearn


def check_if_surrogate_model_is_in_sklearn(name_estimator):

    all_regressors_from_sklearn = [x[0] for x in sklearn.utils.all_estimators(type_filter="regressor")]
    if name_estimator in all_regressors_from_sklearn:
        return True
    else:
        raise ValueError("Surrogate Model is not part of sklearn.")


def select_model_from_sklearn(name_estimator):

    check_if_surrogate_model_is_in_sklearn(name_estimator)

    if name_estimator == "LinearRegression":
        return sklearn.linear_model._base.LinearRegression()
    elif name_estimator == "RandomForestRegressor":
        return sklearn.ensemble._forest.RandomForestRegressor()
