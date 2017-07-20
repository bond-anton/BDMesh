from __future__ import division, print_function
import math as m


def check_if_integer(x, threshold, debug=False):
    L = m.floor(x)
    U = m.ceil(x)
    C = L if abs(L - x) < abs(U - x) else U
    if debug:
        print(x, L, U, C, C - x)
    if abs(C - x) < threshold:
        return True
    else:
        return False
