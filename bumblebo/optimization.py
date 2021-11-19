import sklearn.base

from bumblebo.utils import select_model_from_sklearn

from mbo.algorithm import Algorithm

import opti


class BumbleBO(Algorithm):

    def __init__(self, problem: opti.Problem, params_surrogate=None):
        self.problem: opti.Problem = problem
        if params_surrogate:
            self.params_surrogate: dict = params_surrogate
        else:
            self.params_surrogate: dict = {
                "name": "LinearRegression"
            }

        self.model: sklearn.base.BaseEstimator = select_model_from_sklearn(self.params_surrogate["name"])

        self._fit_model()

    def _fit_model(self) -> None:
        X = self.problem.data[self.problem.inputs.names]
        y = self.problem.data[self.problem.outputs.names]

        self.model.fit(X, y)
