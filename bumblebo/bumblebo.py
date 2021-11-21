from mbo.algorithm import Algorithm
import opti
import pandas as pd
from pymoo.optimize import minimize
import sklearn.base
from typing import Dict

from bumblebo.optimization import choose_optimization_algorithm, SurrogateOptimizationProblem
from bumblebo.learning import select_model_from_sklearn


class BumbleBO(Algorithm):

    def __init__(self, problem: opti.Problem, params_surrogate: dict = None, params_optimization: dict = None):

        self.problem: opti.Problem = problem

        if params_surrogate:
            self.params_surrogate: dict = params_surrogate
        else:
            self.params_surrogate: dict = {
                "name": "LinearRegression"
            }

        if params_optimization:
            self.params_optimization: dict = params_optimization
        else:
            self.params_optimization: dict = {
                "name": "de"
            }

        self.model: Dict[str: sklearn.base.BaseEstimator] = \
            {
                name: select_model_from_sklearn(self.params_surrogate["name"]) for name in self.problem.outputs.names
            }

        self._fit_model()

    def _fit_model(self) -> None:

        X = self.problem.data[self.problem.inputs.names]

        for name in self.problem.outputs.names:
            y = self.problem.data[name]
            self.model[name].fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        predictions = {
            name: self.model[name].predict(X) for name in self.problem.outputs.names
        }
        return pd.DataFrame(predictions)

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        problem = SurrogateOptimizationProblem(self.problem, self.model)
        n_obj = len(self.problem.outputs)
        algorithm = choose_optimization_algorithm(self.params_optimization, n_obj)
        res = minimize(problem, algorithm, seed=73, verbose=True)
        return pd.DataFrame(res.X.reshape(-1,len(self.problem.inputs.names)), columns=self.problem.inputs.names)

    def _choose_optimization_algorithm(self, params_optimization: dict):
        pass
