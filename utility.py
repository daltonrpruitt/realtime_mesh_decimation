# Some utility functions


# https://stackoverflow.com/questions/46335488/how-to-efficiently-find-the-bounding-box-of-a-collection-of-points
def bounding_box(points):
    x_coordinates, y_coordinates, z_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates), min(z_coordinates)), 
            (max(x_coordinates), max(y_coordinates), max(z_coordinates))]
