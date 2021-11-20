import sklearn


def select_model_from_sklearn(name_estimator):

    regressors_sklearn_and_path = {
        x[0]: x[1] for x in sklearn.utils.all_estimators(type_filter="regressor")
    }
    if name_estimator not in regressors_sklearn_and_path:
        raise ValueError("Surrogate Model is not part of sklearn.")

    return regressors_sklearn_and_path[name_estimator]()
