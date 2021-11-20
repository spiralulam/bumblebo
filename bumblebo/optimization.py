import numpy as np
import opti
from pymoo.core.problem import Problem


def build_optimization_problem(problem_formulation: opti.Problem):

    n_var = len(problem_formulation.inputs)
    n_obj = len(problem_formulation.outputs)
    if problem_formulation.constraints:
        n_constr = len(problem_formulation.constraints)
    else:
        n_constr = 0
    xl = np.array(problem_formulation.inputs.bounds.loc["min", :])
    xu = np.array(problem_formulation.inputs.bounds.loc["max", :])

    class SurrogateOptimizationProblem(Problem):
        def __init__(self):
            super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    return SurrogateOptimizationProblem()


def choose_optimization_algorithm():
    pass
