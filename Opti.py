from teneva_bm import *
from teneva_opti import *


class Target(Bm):
    def __init__(self, d, n, objective):
        super().__init__(d=d, n=n)
        self.set_name('problem')
        self.set_desc("""Interesting problem.""")
        self.func = objective

    @property
    def is_opti_max(self):
        return True

    @property
    def is_tens(self):
        return True

    def target(self, i):
        y = self.func(*i)
        return y

