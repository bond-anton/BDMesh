from .Mesh1D cimport Mesh1D

cdef class TreeMesh1D(object):
    """
    Manages the tree of meshes
    """
    cdef:
        dict __tree

    cpdef bint add_mesh(self, Mesh1D mesh, int level=*)
    cpdef int get_mesh_level(self, Mesh1D mesh)
    cpdef dict get_children(self, Mesh1D mesh)
    cpdef bint del_mesh(self, Mesh1D mesh)
    cpdef void remove_coarse_duplicates(self)
    cpdef void recalculate_levels(self)
    cpdef void merge_overlaps_at_level(self, int level)
    cpdef void merge_overlaps(self)
    cpdef void cleanup(self)
    cpdef Mesh1D flatten(self)
