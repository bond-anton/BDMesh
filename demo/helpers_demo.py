from __future__ import division, print_function
from BDMesh._helpers import check_if_integer


print(check_if_integer(0.99))
print(check_if_integer(1.0))
print(check_if_integer(0.99, 0.02))
