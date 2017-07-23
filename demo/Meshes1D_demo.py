from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt

from BDMesh import TreeMeshUniform1D, MeshUniform1D


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
            ax.plot(mesh.physical_nodes, np.ones(mesh.num) * level, colors[level] + styles[i] + 'o')
    ax.set_ylim([-1, max(mesh_tree.Tree.keys()) + 1])
    ax.grid()
    if show:
        plt.show()

root_mesh = MeshUniform1D(0.0, 10.0, 1, boundary_condition_1=1, boundary_condition_2=0)
print(root_mesh.physical_nodes)
child_mesh_1_1 = MeshUniform1D(0.0, 9.0, 0.5, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_1_2 = MeshUniform1D(3.0, 17.0, 0.5, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_1_3 = MeshUniform1D(6.0, 8.0, 0.5, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_2_1 = MeshUniform1D(1.0, 2.0, 0.25, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_2_2 = MeshUniform1D(2.0, 5.0, 0.25, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_3_1 = MeshUniform1D(1.0, 1.5, 0.125, boundary_condition_1=1, boundary_condition_2=0)
Meshes = TreeMeshUniform1D(root_mesh)
Meshes.add_mesh(child_mesh_1_1)
print(Meshes.Tree)
Meshes.add_mesh(child_mesh_1_2)
Meshes.add_mesh(child_mesh_1_3)
print(Meshes.Tree)
Meshes.add_mesh(child_mesh_2_1)
Meshes.add_mesh(child_mesh_2_2)
Meshes.add_mesh(child_mesh_3_1)

print(Meshes.Tree)
plot_tree(Meshes)
