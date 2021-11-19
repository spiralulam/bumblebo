from bumblebo import BumbleBO
from opti.problems import Zakharov


def test_continuous_single_objective_unconstrained():

    test_problem = Zakharov(n_inputs=3)
    test_problem.create_initial_data(n_samples=10)

    params_surrogate = {
        "name": "RandomForestRegressor"
    }

    bbo = BumbleBO(problem=test_problem, params_surrogate=params_surrogate)


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
