import sympy
import matplotlib.pyplot as plt
from frame_drawer import FrameDrawer
import numpy as np
from collections import deque


class Kinematics:
    def __init__(self, dh_root):
        self._dh_root = dh_root
        self._coordinates = []
        self._coordinates_t = []
        self._d_coordinates = []
        self._d_coordinates_t = []
        self._dd_coordinates = []
        self._dd_coordinates_t = []

        self._subs_q2qt = []
        self._subs_dq2dqt = []
        self._subs_ddq2ddqt = []
        self._subs_qt2q = []
        self._subs_dqt2dq = []
        self._subs_ddqt2ddq = []

    def _cal_transfmat_iter(self, node):
        prev_link = node.get_prev_link()
        succ_link = node.get_succ_link()

        # none root link
        if prev_link is not None:
            # transformation matrix of frame
            node.cal_motion_params(prev_link.T_0n)
            for c, ct, dc, dct, ddc, ddct in zip(node._coordinates, node._coordinates_t, node._d_coordinates, node._d_coordinates_t, node._dd_coordinates, node._dd_coordinates_t):
                if c not in self._coordinates:
                    self._coordinates.append(c)
                    self._coordinates_t.append(ct)
                    self._d_coordinates.append(dc)
                    self._d_coordinates_t.append(dct)
                    self._dd_coordinates.append(ddc)
                    self._dd_coordinates_t.append(ddct)

            print("node: {}".format(node.get_num()))

        # last link
        if len(succ_link) is 0:
            return

        for succ in succ_link:
            self._cal_transfmat_iter(succ)

    def cal_transfmats(self):
        self._cal_transfmat_iter(self._dh_root)

        self._subs_q2qt = [(q, qt) for q, qt in zip(self._coordinates, self._coordinates_t)]
        self._subs_dq2dqt = [(dq, dqt) for dq, dqt in zip(self._d_coordinates, self._d_coordinates_t)]
        self._subs_ddq2ddqt = [(ddq, ddqt) for ddq, ddqt in zip(self._dd_coordinates, self._dd_coordinates_t)]

        self._subs_qt2q = [(qt, q) for q, qt in zip(self._coordinates, self._coordinates_t)]
        self._subs_dqt2dq = [(dqt, dq) for dq, dqt in zip(self._d_coordinates, self._d_coordinates_t)]
        self._subs_ddqt2ddq = [(ddqt, ddq) for ddq, ddqt in zip(self._dd_coordinates, self._dd_coordinates_t)]


    def get_coordinates(self):
        return self._coordinates
