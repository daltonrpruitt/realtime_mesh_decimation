import utility
import pywavefront
import numpy as np

def get_vertex_cell_indices_ids(vertices = None, resolution=None):
    if vertices is None or resolution is None:
        print("Cell ID generation requires both vertices and resolution")
        exit()
    bbox = utility.bounding_box(vertices) # Makes bounding box
    pos_verts = utility.positive_vertices(vertices,bbox_in=bbox)
    avg = [(bbox[0][i] + bbox[1][i]) / 2.0 for i in range(3) ]
    scale = [(bbox[1][i] - bbox[0][i]) for i in range(3) ] # / 2.0
    mesh_min = bbox[0]

    #print("Min =", mesh_min)
    #print("Avg =", avg, "  Scale =",scale)
    '''
    x, y, z = zip(*pos_verts)
    print("Min =", min(x), min(y), min(z))
    print("Max =", max(x), max(y), max(z))
    print("Scale = ", scale)
    '''

    scaled_verts = []
    for vert in pos_verts:
        scaled_verts.append([vert[i] * resolution**2 / (scale[i]+0.00001) for i in range(3)])

    '''
    x, y, z = zip(*scaled_verts)
    print("Min =", min(x), min(y), min(z))
    print("Max =", max(x), max(y), max(z))
    '''
    vertex_cluster_cell_indices = []
    vertex_cluster_cell_ids = []
    for vert in scaled_verts:
        indices = [vert[i] // resolution for i in range(3)]
        id = indices[0] + indices[1] * resolution + indices[2] * resolution**2

        vertex_cluster_cell_indices.append(indices)
        vertex_cluster_cell_ids.append(id)

    if len(vertex_cluster_cell_indices) != len(vertices) or len(vertex_cluster_cell_ids) != len(vertices) :
        print("Error: Cluster IDs/Indices generated is not same length as vertices input!")
        exit()

    return vertex_cluster_cell_indices, vertex_cluster_cell_ids


def test_cluster_cell_id_gen():
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    verts = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.0, 1.0, -1.0],
        ]
    indices, ids = get_vertex_cell_indices_ids(vertices=verts,resolution=3)
    print(indices)
    print(ids)

    xdata = [v[0] for v in verts]
    ydata = [v[1] for v in verts]
    zdata = [v[2] for v in verts]

    print(xdata, ydata, zdata)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_trisurf(xdata, ydata, zdata, cmap='viridis', edgecolor='none')
    fig.show()
    input()



if __name__ == "__main__":
    test_cluster_cell_id_gen()