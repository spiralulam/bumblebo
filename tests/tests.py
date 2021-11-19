from bumblebo import BumbleBO
from opti.problems import Zakharov


def test_continuous_single_objective_unconstrained():

    test_problem = Zakharov(n_inputs=3)
    test_problem.create_initial_data(n_samples=10)

    bbo = BumbleBO(problem=test_problem)
