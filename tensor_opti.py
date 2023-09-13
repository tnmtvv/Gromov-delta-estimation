from teneva_bm import *
from teneva_opti import *


class Target(Bm):
    def init(self):
        super().init(d=3, n=500)
        self.set_name('problem')
        self.set_desc("""Interesting problem.""")

    @property
    def is_opti_max(self):
        return True

    @property
    def is_tens(self):
        return True

    def target(self, i):
        y = obj_func(*i)
        return y

targ = Target()
opti = OptiTensProtes(targ, m=1.E+5, log_info=True, log_file=True)
opti.run()
opti.save()