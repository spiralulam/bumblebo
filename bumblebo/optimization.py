import numpy as np
import opti
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.factory import get_algorithm
import sklearn


class SurrogateOptimizationProblem(Problem):

    def __init__(self, problem_formulation: opti.Problem, surrogate_model: sklearn.base.BaseEstimator):

        self.problem_formulation = problem_formulation
        self.surrogate_model = surrogate_model

        n_var = len(problem_formulation.inputs)
        n_obj = len(problem_formulation.outputs)
        if problem_formulation.constraints:
            n_constr = len(problem_formulation.constraints)
        else:
            n_constr = 0
        xl = np.array(problem_formulation.inputs.bounds.loc["min", :])
        xu = np.array(problem_formulation.inputs.bounds.loc["max", :])

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.surrogate_model.predict(x)
        if self.problem_formulation.constraints:
            out["G"] = np.array(
                [
                constraint(pd.DataFrame(x, columns=self.problem_formulation.inputs.names))
                for constraint in self.problem_formulation.constraints
                ]
            ).reshape(-1, 1)


def choose_optimization_algorithm(params_optimization: dict):
    name_algorithm = params_optimization["name"]
    return get_algorithm(name_algorithm)
