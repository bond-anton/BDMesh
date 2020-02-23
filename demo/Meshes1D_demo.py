import numpy as np
from matplotlib import pyplot as plt

from BDMesh import TreeMesh1DUniform, Mesh1DUniform


def plot_tree(mesh_tree, ax=None):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k',
              'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k']
    styles = ['-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-',
              '-', ':', '--', '-', ':', '--', '-', '-', ':', '--', '-', ':', '--', '-']
    show = False
    if ax is None:
        _, ax = plt.subplots()
        show = True
    for level in mesh_tree.tree.keys():
        for i, mesh in enumerate(mesh_tree.tree[level]):
            ax.plot(mesh.physical_nodes, np.ones(mesh.num) * level, colors[level] + styles[i] + 'o')
    ax.set_ylim([-1, max(mesh_tree.tree.keys()) + 1])
    ax.grid()
    if show:
        plt.show()

root_mesh = Mesh1DUniform(0.0, 10.0, physical_step=1.0, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_1_1 = Mesh1DUniform(0.0, 9.0, physical_step=0.5, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_1_2 = Mesh1DUniform(3.0, 17.0, physical_step=0.5, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_1_3 = Mesh1DUniform(6.0, 8.0, physical_step=0.5, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_2_1 = Mesh1DUniform(1.0, 2.0, physical_step=0.25, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_2_2 = Mesh1DUniform(2.0, 5.0, physical_step=0.25, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_3_1 = Mesh1DUniform(1.0, 1.5, physical_step=0.125, boundary_condition_1=1, boundary_condition_2=0)
Meshes = TreeMesh1DUniform(root_mesh, aligned=False)
Meshes.add_mesh(child_mesh_1_1)
print(Meshes.tree)
Meshes.add_mesh(child_mesh_1_2)
Meshes.add_mesh(child_mesh_1_3)
print(Meshes.tree)
Meshes.add_mesh(child_mesh_2_1)
Meshes.add_mesh(child_mesh_2_2)
Meshes.add_mesh(child_mesh_3_1)

print(Meshes.tree)
print(root_mesh.physical_nodes)
print(root_mesh.physical_step, np.gradient(root_mesh.physical_nodes))
plot_tree(Meshes)
Meshes.crop = [1, 1]
Meshes.trim()
plot_tree(Meshes)
Meshes.crop = [1, 1]
Meshes.trim()
plot_tree(Meshes)
