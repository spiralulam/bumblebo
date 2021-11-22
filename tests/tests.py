from bumblebo import BumbleBO
from opti.problems import Zakharov, Zakharov_Constrained, Detergent
import sklearn


def test_continuous_single_objective_unconstrained():

    test_problem = Zakharov(n_inputs=3)
    test_problem.create_initial_data(n_samples=10)

    params_surrogate = {
        "name": "RandomForestRegressor"
    }

    params_optimization = {
        "name": "ga"
    }

    bbo = BumbleBO(problem=test_problem, params_surrogate=params_surrogate, params_optimization=params_optimization)

    # This function returns None, if the model is fitted and raises an NotFittedError otherwise
    models_fitted = [sklearn.utils.validation.check_is_fitted(m) for m in bbo.model.values()]
    assert all(v is None for v in models_fitted)

    X_pred = bbo.problem.data[bbo.problem.inputs.names]
    y_pred = bbo.predict(X_pred)
    assert len(y_pred) == len(X_pred)

    X_next = bbo.propose(n_proposals=1)

    assert X_next.shape == (1, 3)


def test_continuous_single_objective_constrained():

    test_problem = Zakharov_Constrained(n_inputs=3)
    test_problem.create_initial_data(n_samples=10)

    params_surrogate = {
        "name": "RandomForestRegressor"
    }
    params_optimization = {
        "name": "ga"
    }

    bbo = BumbleBO(problem=test_problem, params_surrogate=params_surrogate, params_optimization=params_optimization)

    # This function returns None, if the model is fitted and raises an NotFittedError otherwise
    models_fitted = [sklearn.utils.validation.check_is_fitted(m) for m in bbo.model.values()]
    assert all(v is None for v in models_fitted)

    X_pred = bbo.problem.data[bbo.problem.inputs.names]
    y_pred = bbo.predict(X_pred)
    assert len(y_pred) == len(X_pred)

    X_next = bbo.propose(n_proposals=1)

    assert X_next.shape == (1, 3)


def test_multi_objective_constrained():

    test_problem = Detergent()
    test_problem.create_initial_data(n_samples=50)

    params_surrogate = {
        "name": "RandomForestRegressor"
    }
    params_optimization = {
        "name": "ctaea"
    }

    bbo = BumbleBO(problem=test_problem, params_surrogate=params_surrogate, params_optimization=params_optimization)

    # This function returns None, if the model is fitted and raises an NotFittedError otherwise
    models_fitted = [sklearn.utils.validation.check_is_fitted(m) for m in bbo.model.values()]
    assert all(v is None for v in models_fitted)

    X_pareto = bbo.predict_pareto_front()

    assert X_pareto.shape[1] == len(test_problem.inputs)
    assert X_pareto.shape[0] > 1


def test_wrong_surrogate_model():

    test_problem = Zakharov(n_inputs=3)
    test_problem.create_initial_data(n_samples=10)

    params_surrogate = {
        "name": "BumbleRegressor"
    }

    try:
        BumbleBO(problem=test_problem, params_surrogate=params_surrogate)
    except ValueError:
        assert True
    else:
        assert False


def test_all_sklearn_regressors():

    test_problem = Zakharov(n_inputs=3)
    test_problem.create_initial_data(n_samples=10)

    all_sklearn_regressors = [x[0] for x in sklearn.utils.all_estimators(type_filter="regressor")]

    regressors_with_problems = ["IsotonicRegression", "MultiOutputRegressor", "RegressorChain", "StackingRegressor",
                                "VotingRegressor", "MultiTaskElasticNet", "MultiTaskElasticNetCV", "MultiTaskLasso",
                                "MultiTaskLassoCV", "PLSCanonical", "PLSRegression", "CCA"]

    for regressor in all_sklearn_regressors:

        if regressor not in regressors_with_problems:

            params_surrogate = {
                "name": regressor
            }

            bbo = BumbleBO(problem=test_problem, params_surrogate=params_surrogate)

            # This function returns None, if the model is fitted and raises an NotFittedError otherwise
            models_fitted = [sklearn.utils.validation.check_is_fitted(m) for m in bbo.model.values()]
            assert all(v is None for v in models_fitted)

            X_pred = bbo.problem.data[bbo.problem.inputs.names]
            y_pred = bbo.predict(X_pred)

            assert len(y_pred) == len(X_pred)
