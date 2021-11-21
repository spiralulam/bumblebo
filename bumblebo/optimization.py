import numpy as np
import opti
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.factory import get_algorithm, get_reference_directions
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
        out["F"] = np.array(
            [self.surrogate_model[name].predict(x) for name in self.problem_formulation.outputs.names]
        ).reshape(-1, self.n_obj)
        if self.problem_formulation.constraints:
            out["G"] = np.array(
                [
                constraint(pd.DataFrame(x, columns=self.problem_formulation.inputs.names))
                for constraint in self.problem_formulation.constraints
                ]
            ).reshape(-1, 1)


def choose_optimization_algorithm(params_optimization: dict, n_obj: int):
    name_algorithm = params_optimization["name"]

    # These multi-objective optimization algorithms need reference directions as inputs.
    # I know that this hard coded list is not nice, but I didn't find a better solution.
    if name_algorithm in ["ctaea", "moead", "unsga3", "nsga3"]:
        ref_dirs = get_reference_directions("energy", n_dim=n_obj, n_points=n_obj, seed=73)
        return get_algorithm(name_algorithm, ref_dirs=ref_dirs)
    else:
        return get_algorithm(name_algorithm)
