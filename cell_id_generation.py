import utility
import pywavefront
import numpy as np

def get_vertex_cell_indices_ids(vertices = None, resolution=None):
    if vertices is None or resolution is None:
        print("Cell ID generation requires both vertices and resolution")
        exit()
    bbox = utility.bounding_box(vertices) # Makes bounding box

    avg = [(bbox[0][i] + bbox[1][i]) / 2.0 for i in range(3) ]
    scale = [(bbox[1][i] - bbox[0][i]) for i in range(3) ] # / 2.0
    mesh_min = bbox[0]

    #print("Min =", mesh_min)
    #print("Avg =", avg, "  Scale =",scale)
    pos_verts = []
    for vert in vertices:
        pos_verts.append([vert[i] + -mesh_min[i] for i in range(3)])
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

'''
resolution = 10
obj_mesh = pywavefront.Wavefront("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj", collect_faces=True)
vertices = np.array(obj_mesh.vertices, dtype='f4')
bbox = utility.bounding_box(vertices) # Makes bounding box

# Pass 1 - Step 1 : Determine Cell ID
#      a. Shrink model to -1, 1
avg = [(bbox[0][i] + bbox[1][i]) / 2.0 for i in range(3) ]
scale = [(bbox[1][i] - bbox[0][i]) for i in range(3) ] # / 2.0
mesh_min = bbox[0]
print("Min =", mesh_min)
print("Avg =", avg, "  Scale =",scale)
pos_verts = []
for vert in vertices:
    pos_verts.append([vert[i] + -mesh_min[i] for i in range(3)])
x, y, z = zip(*pos_verts)
print("Min =", min(x), min(y), min(z))
print("Max =", max(x), max(y), max(z))
print("Scale = ", scale)

scaled_verts = []
for vert in pos_verts:
    scaled_verts.append([vert[i] * resolution**2 / (scale[i]+0.00001) for i in range(3)])

x, y, z = zip(*scaled_verts)
print("Min =", min(x), min(y), min(z))
print("Max =", max(x), max(y), max(z))

cell_ids = {}
for vert in scaled_verts:
    indices = [vert[i] // resolution for i in range(3)]
    id = indices[0] + indices[1] * resolution + indices[2] * resolution**2
    if id in cell_ids:
        cell_ids[id] += 1
    else:
        cell_ids[id] = 1

print(len(cell_ids.items()))
print("Cell IDs min,max =",min(cell_ids), max(cell_ids))
import operator
print("Most common IDs: ", sorted(cell_ids.items(), key=operator.itemgetter(1), reverse=True)[:10])
print("Total vertices found =", sum(cell_ids.values()))

#float x_range = resolution * resolution;
#float y_range = resolution * 4;
x_range = resolution ** 2
y_range = resolution * 4

output_pix_pos = []
output_pos = []
#gl_Position = vec4( (2.0 * mod(cell_id[0], x_range) - x_range)/x_range, -1.0 * (2.0 * 4.0 * trunc(cell_id[0] / (resolution * resolution)) - y_range) / y_range , 0.0, 0.0);
for id in cell_ids.keys():
    x = id % x_range
    y =  4 * id // x_range
    output_pix_pos.append((x, y))

    xpos = (2.0 *  x - x_range)/x_range
    ypos = -1.0 * (2.0 * y - y_range) / y_range 
    output_pos.append((xpos, ypos))

x, y = zip(*output_pix_pos)
print("Output_pix_pos : min, max =",min(x), min(y), "|",max(x),max(y))

print(output_pos[:5])
#print(len(output_pos))
x, y = zip(*output_pos)
print("Output_pos : min, max =",min(x), min(y), "|",max(x),max(y))
'''