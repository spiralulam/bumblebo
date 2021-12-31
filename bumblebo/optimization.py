from typing import Dict

import numpy as np
import opti
import pandas as pd
import sklearn
from pymoo.core.problem import Problem as PymooProblem
from pymoo.factory import get_algorithm, get_reference_directions


class SurrogateOptimizationProblem(PymooProblem):
    def __init__(
        self, problem: opti.Problem, surrogate: Dict[str, sklearn.base.BaseEstimator]
    ):

        self.problem = problem
        self.n_obj = len(problem.objectives)
        self.surrogate = surrogate

        super().__init__(
            n_var=len(problem.inputs),
            n_obj=len(problem.objectives),
            n_constr=len(problem.constraints) if problem.constraints else 0,
            xl=problem.inputs.bounds.loc["min"].values,
            xu=problem.inputs.bounds.loc["max"].values,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array(
            [self.surrogate[name].predict(x) for name in self.problem.outputs.names]
        ).reshape(-1, self.n_obj)
        if self.problem.constraints:
            X = pd.DataFrame(x, columns=self.problem.inputs.names)
            out["G"] = self.problem.constraints(X).values


def choose_optimization_algorithm(params_optimization: dict, n_obj: int = 1):
    name_algorithm = params_optimization["name"]
    # These multi-objective optimization algorithms need reference directions as inputs.
    # I know that this hard coded list is not nice, but I didn't find a better solution.
    if name_algorithm in ["ctaea", "moead", "unsga3", "nsga3"]:
        ref_dirs = get_reference_directions(
            "energy", n_dim=n_obj, n_points=n_obj ** 2, seed=73
        )
        return get_algorithm(name_algorithm, ref_dirs=ref_dirs)
    else:
        return get_algorithm(name_algorithm)
