from mbo.algorithm import Algorithm

import opti


class BumbleBO(Algorithm):

    def __init__(self, problem: opti.Problem):
        self.problem: opti.Problem = problem
