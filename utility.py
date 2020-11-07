# Some utility functions


# https://stackoverflow.com/questions/46335488/how-to-efficiently-find-the-bounding-box-of-a-collection-of-points
def bounding_box(points):
    x_coordinates, y_coordinates, z_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates), min(z_coordinates)), 
            (max(x_coordinates), max(y_coordinates), max(z_coordinates))]


def positive_vertices(vertices, bbox_in=None):
    if bbox_in is None:
        bbox_in = bounding_box(vertices)
    mesh_min = bbox_in[0]
    pos_verts = []
    for vert in vertices:
        pos_verts.append([vert[i] + -mesh_min[i] for i in range(3)])
    return pos_verts