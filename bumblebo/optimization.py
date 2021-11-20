import numpy as np
import opti
from pymoo.core.problem import Problem


class SurrogateOptimizationProblem(Problem):
    def __init__(self, n_var: int, n_obj: int, n_constr: int, xl: np.ndarray, xu: np.ndarray):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        pass
        # out["F"] = connect with problem.f method from opti
        # out["G"] = connect with constraints.satisfied methods from opti


def build_optimization_problem(problem_formulation: opti.Problem):

    n_var = len(problem_formulation.inputs)
    n_obj = len(problem_formulation.outputs)
    if problem_formulation.constraints:
        n_constr = len(problem_formulation.constraints)
    else:
        n_constr = 0
    xl = np.array(problem_formulation.inputs.bounds.loc["min", :])
    xu = np.array(problem_formulation.inputs.bounds.loc["max", :])

    return SurrogateOptimizationProblem(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)


def choose_optimization_algorithm():
    pass
