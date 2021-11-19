from bumblebo import BumbleBO
from opti.problems import Zakharov
import sklearn


def test_continuous_single_objective_unconstrained():

    test_problem = Zakharov(n_inputs=3)
    test_problem.create_initial_data(n_samples=10)

    params_surrogate = {
        "name": "RandomForestRegressor"
    }

    params_optimization = {
        "name": "GA"
    }

    bbo = BumbleBO(problem=test_problem, params_surrogate=params_surrogate, params_optimization=params_optimization)

    # This function returns None, if the model is fitted and raises an NotFittedError otherwise
    assert sklearn.utils.validation.check_is_fitted(bbo.model) is None

    X_pred = bbo.problem.data[bbo.problem.inputs.names]
    y_pred = bbo.predict(X_pred)
    assert len(y_pred) == len(X_pred)

    bbo.propose()


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
