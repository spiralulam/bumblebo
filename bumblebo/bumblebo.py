from mbo.algorithm import Algorithm
import opti
import pandas as pd
import sklearn.base

from bumblebo.optimization import build_optimization_problem, choose_optimization_algorithm
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
            self.params_optimization: dict = params_surrogate
        else:
            self.params_optimization: dict = {
                "name": "DE"
            }

        self.model: sklearn.base.BaseEstimator = select_model_from_sklearn(self.params_surrogate["name"])

        self._fit_model()

    def _fit_model(self) -> None:

        X = self.problem.data[self.problem.inputs.names]
        y = self.problem.data[self.problem.outputs.names]

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:

        return self.model.predict(X)

    def propose(self, n_proposals: int = 1) -> pd.DataFrame:
        problem = build_optimization_problem(self.problem)
        algorithm = choose_optimization_algorithm()
        # res = pm.optimize.minimize(problem, algorithm, seed=73, verbose=True)

    def _choose_optimization_algorithm(self, params_optimization: dict):
        pass
