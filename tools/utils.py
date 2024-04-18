import sympy
import numpy as np
import cloudpickle as pickle
import os.path
import os
import errno
import csv


def new_sym(name):
    return sympy.symbols(name, real=True)


def vec2so3(vec):
    return sympy.Matrix([[0,        -vec[2],    vec[1]],
                         [vec[2],   0,          -vec[0]],
                         [-vec[1],  vec[0],     0]])


def so32vec(mat):
    return sympy.Matrix([[mat[2, 1]],
                         [mat[0, 2]],
                         [mat[1, 0]]])


def inertia_vec2tensor(vec):
    return sympy.Matrix([[vec[0], vec[1], vec[2]],
                         [vec[1], vec[3], vec[4]],
                         [vec[2], vec[4], vec[5]]])


def inertia_tensor2vec(I):
    return [I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2]]


def tranlation_transfmat(v):
    return sympy.Matrix([[1, 0, 0, v[0]],
                        [0, 1, 0, v[1]],
                        [0, 0, 1, v[2]],
                        [0, 0, 0, 1]])


def ml2r(m, l):
    return sympy.Matrix(l) / m


def Lmr2I(L, m, r):
    return sympy.Matrix(L - m * vec2so3(r).transpose() * vec2so3(r))



def save_data(folder, name, data):
    model_file = folder + name + '.pkl'

    if not os.path.exists(os.path.dirname(model_file)):
        try:
            os.makedirs(os.path.dirname(model_file))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(model_file, 'w+') as f:
        pickle.dump(data, f)

def save_csv_data(folder, name, data):
    with open(folder + name + '.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        for i in range(np.size(data, 0) - 10):
            wr.writerow(data[i])

def load_data(folder, name):
    model_file = folder + name + '.pkl'
    if os.path.exists(model_file):
        data = pickle.load(open(model_file, 'rb'))
        return data
    else:
        raise Exception("No {} can be found!".format(model_file))