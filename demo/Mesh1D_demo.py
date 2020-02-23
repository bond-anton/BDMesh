import numpy as np
from BDMesh.Mesh1D import Mesh1D


MyMesh = Mesh1D(0.0, 10.0, 0.0, 0.0)

print(MyMesh)
print(MyMesh.physical_boundary_1)

MyMesh.physical_boundary_1 = 2
print(MyMesh.physical_boundary_1)

MyMesh2 = Mesh1D(2.0, 10.0, 0.0, 0.0)

print(MyMesh)
print(MyMesh2)

print(MyMesh == MyMesh2)
print(MyMesh == 2)

print(MyMesh.local_nodes)
MyMesh.local_nodes = np.linspace(0.0, 1.0, num=11, endpoint=True, dtype=np.float64)
print(MyMesh.local_nodes)
print(MyMesh.num)
print(MyMesh.physical_nodes)


