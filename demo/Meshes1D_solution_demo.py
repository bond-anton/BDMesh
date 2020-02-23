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
        _, ax = plt.subplots(2, 1)
        show = True
    for level in mesh_tree.tree.keys():
        for i, mesh in enumerate(mesh_tree.tree[level]):
            ax[0].plot(mesh.physical_nodes, np.ones(mesh.num) * level, colors[level] + styles[i] + 'o')
            ax[1].plot(mesh.physical_nodes, mesh.solution, colors[level] + styles[i] + 'o')
    ax[0].set_ylim([-1, max(mesh_tree.tree.keys()) + 1])
    ax[0].grid()
    ax[1].grid()
    if show:
        plt.show()


root_mesh = Mesh1DUniform(0.0, 10.0, physical_step=1.0, boundary_condition_1=1, boundary_condition_2=0)
root_mesh.solution = np.asarray(root_mesh.physical_nodes) ** 2

child_mesh_1_1 = Mesh1DUniform(0.0, 9.0, physical_step=0.5, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_1_1.solution = np.asarray(child_mesh_1_1.physical_nodes) ** 2

child_mesh_1_2 = Mesh1DUniform(3.0, 17.0, physical_step=0.5, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_1_2.solution = np.asarray(child_mesh_1_2.physical_nodes) ** 2

child_mesh_1_3 = Mesh1DUniform(6.0, 8.0, physical_step=0.5, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_1_3.solution = np.asarray(child_mesh_1_3.physical_nodes) ** 2

child_mesh_2_1 = Mesh1DUniform(1.0, 2.0, physical_step=0.25, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_2_1.solution = np.asarray(child_mesh_2_1.physical_nodes) ** 2

child_mesh_2_2 = Mesh1DUniform(2.0, 5.0, physical_step=0.25, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_2_2.solution = np.asarray(child_mesh_2_2.physical_nodes) ** 2

child_mesh_3_1 = Mesh1DUniform(1.0, 3.5, physical_step=0.125, boundary_condition_1=1, boundary_condition_2=0)
child_mesh_3_1.solution = np.asarray(child_mesh_3_1.physical_nodes) ** 2

Meshes = TreeMesh1DUniform(root_mesh, aligned=False)
Meshes.add_mesh(child_mesh_1_2)
Meshes.add_mesh(child_mesh_1_1)
Meshes.add_mesh(child_mesh_1_3)

Meshes.add_mesh(child_mesh_2_1)
Meshes.add_mesh(child_mesh_2_2)
Meshes.add_mesh(child_mesh_3_1)

# plot_tree(Meshes)
# Meshes.crop = [1, 1]
# Meshes.trim()
# plot_tree(Meshes)
# Meshes.crop = [2, 2]
# Meshes.trim()

flattened = Meshes.flatten()

query_points_x = np.arange(1, 6) * np.pi / 1
query_points_y = np.asarray(Meshes.interpolate_solution(query_points_x))

_, ax = plt.subplots(3, 1)
plot_tree(Meshes, ax)
ax[2].plot(flattened.physical_nodes, flattened.solution, 'r-o')
ax[1].plot(query_points_x, query_points_y, '-x')
ax[2].plot(query_points_x, query_points_y, '-x')
ax[2].grid(True)
for query_point_x, query_point_y in zip(query_points_x, query_points_y):
    ax[0].axvline(x=query_point_x)
    ax[1].axvline(x=query_point_x)
    # ax[1].axhline(y=query_point_y)
    ax[2].axvline(x=query_point_x)
    # ax[2].axhline(y=query_point_y)

plt.show()
