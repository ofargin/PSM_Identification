import sympy
from sympy import lambdify
from sympy.utilities.iterables import flatten
import numpy as np
from collections import deque
from utils import vec2so3, new_sym
from dyn_param_dep import find_dyn_parm_deps
from sympy import pprint
import time
import copy
import math
import csv


class Dynamics:
    def __init__(self, rbt_def, geom, g=[0, 0, -9.81], verbose=False):
        self.rbt_def = rbt_def
        self.geom = geom
        self._g = np.matrix(g)
