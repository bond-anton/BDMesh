from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDMesh import Uniform1DMeshesTree, UniformMesh1D


def plot_tree(mesh_tree, ax=None):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k',
              'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
    styles = ['-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-']
    show = False
    if ax is None:
        _, ax = plt.subplots()
        show = True
    for level in mesh_tree.Tree.keys():
        for i, mesh in enumerate(mesh_tree.Tree[level]):
            ax.plot(mesh.to_phys(mesh.local_nodes), np.ones(mesh.num) * level, colors[level] + styles[i] + 'o')
    ax.set_ylim([-1, max(mesh_tree.Tree.keys()) + 1])
    ax.grid()
    if show:
        plt.show()

root_mesh = UniformMesh1D(0.0, 10.0, 1, bc1=1, bc2=0)
child_mesh_1_1 = UniformMesh1D(0.0, 9.0, 0.5, bc1=1, bc2=0)
#child_mesh_1_2 = UniformMesh1D(3.0, 17.0, 0.5, bc1=1, bc2=0)
child_mesh_1_3 = UniformMesh1D(6.0, 8.0, 0.5, bc1=1, bc2=0)
child_mesh_2_1 = UniformMesh1D(1.0, 2.0, 0.25, bc1=1, bc2=0)
child_mesh_2_2 = UniformMesh1D(2.0, 5.0, 0.25, bc1=1, bc2=0)
child_mesh_3_1 = UniformMesh1D(1.0, 1.5, 0.125, bc1=1, bc2=0)
Meshes = Uniform1DMeshesTree(root_mesh)
Meshes.add_mesh(child_mesh_1_1)
print(Meshes.Tree)
#Meshes.add_mesh(child_mesh_1_2)
Meshes.add_mesh(child_mesh_1_3)
print(Meshes.Tree)
Meshes.add_mesh(child_mesh_2_1)
Meshes.add_mesh(child_mesh_2_2)
Meshes.add_mesh(child_mesh_3_1)

print(Meshes.Tree)
plot_tree(Meshes)
