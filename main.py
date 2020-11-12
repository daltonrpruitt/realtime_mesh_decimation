'''
    Dalton Winans-Pruitt  dp987
    Based on Hello World example from https://github.com/moderngl/moderngl/tree/master/examples
    Using Compute Shaders for mesh decimation
'''

import moderngl
from moderngl_window import screenshot
import numpy as np
from window_setup import BasicWindow  # imports moderngl_window

from moderngl import Uniform
import time
import trimesh
import pywavefront
import utility
import transformations as transf
import os
from cell_id_generation import get_vertex_cell_indices_ids
import box, triangle


# Directly from moderngl compute_shader.py example : https://github.com/moderngl/moderngl/blob/master/examples/compute_shader.py
def source(uri, consts):
    ''' read gl code '''
    with open(uri, 'r') as fp:
        content = fp.read()

    # feed constant values
    for key, value in consts.items():
        content = content.replace(f"%%{key}%%", str(value))
    return content

renderonly = False  

debug = False
resolution = 20

float_to_int_scaling_factor = 2**13
use_box = False
use_triangle = False

if use_box :
    box = box.Box()
    vertices = np.array(utility.positive_vertices(box.vertices), dtype=np.float32)
    indices = np.array(box.indicies, dtype=np.int32)
    print("Vertex Count:",len(vertices),"  Tri Count:",len(indices))

elif use_triangle:
    tri = triangle.Triangle()
    vertices = np.array(utility.positive_vertices(tri.vertices), dtype=np.float32)
    indices = np.array(tri.indicies, dtype=np.int32)


else:
    st = time.time()
    #self.obj_mesh = trimesh.exchange.load.load("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj")
    obj_mesh = pywavefront.Wavefront("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj", collect_faces=True)
    print("Loading mesh took {:.2f} s".format(time.time()-st))
    #vertices = np.array(obj_mesh.vertices, dtype='f4')

    # indices are a list of groups of 3 indices into the vertex array, each group representing 1 triangle
    indices = np.array(obj_mesh.mesh_list[0].faces,dtype=np.int32)
    # Using the positive transformed version for ease of distinguishing valid points
    vertices = np.array(utility.positive_vertices(obj_mesh.vertices),dtype="f4")

# Solves an issue with the compute shader not getting the indices correct?
indices = np.append(indices, np.zeros(shape=(len(indices),1),dtype=np.int32), axis=1)
vertices = np.append(vertices, np.zeros(shape=(len(vertices),1),dtype=np.float32), axis=1)

        
bbox = utility.bounding_box(vertices[:,:3]) # Makes bounding box
if debug:
    print("Input Vertices:")
    print("\tMin,max of x:",min(vertices[:,0]),",",max(vertices[:,0]))
    print("\tMin,max of y:",min(vertices[:,1]),",",max(vertices[:,1]))
    print("\tMin,max of z:",min(vertices[:,2]),",",max(vertices[:,2]))
    #print(vertices)
    #print(indices)

if not renderonly:

    st = time.time()
    vertex_cluster_indices, vertex_cluster_ids = get_vertex_cell_indices_ids(vertices=vertices[:,:3], resolution=resolution)
    print("Generating vertex cluster indicies and IDs took {:.5f} s".format(time.time()-st))
    vertex_cluster_ids = np.array(vertex_cluster_ids, dtype=np.int32)

    # Note Cluster ID Generation works for the box, at least!
    if debug:
        print("Cluster IDs/ Indices")
        print(vertex_cluster_ids)
        print(vertex_cluster_indices)

    #print(vertex_cluster_ids[:30])
    print("No. of Tris:", len(indices))

    shader_constants = {
        "NUM_VERTS": len(vertices), 
        "NUM_TRIS" : len(indices), 
        "X": 1,
        "Y": 1, 
        "Z": 1
    }

    #size = len(indices)
    size = resolution**3


    # Standalone since not doing any rendering yet
    fp_context = moderngl.create_standalone_context(require=430)
    first_pass_comp_shader = fp_context.compute_shader(source("shaders/firstpass.comp", shader_constants))
    print("Successfully compiled 1st-pass compute shader!")



    # Create/bind vertex/index data
    vertex_buffer = fp_context.buffer(vertices.astype("f4").tobytes())
    vertex_buffer.bind_to_storage_buffer(binding=0)
    index_buffer = fp_context.buffer(indices.astype("i4").tobytes())
    index_buffer.bind_to_storage_buffer(binding=1)

    #print(indices)
    #print(np.frombuffer(index_buffer.read(),dtype=np.int32))

    cluster_id_buffer = fp_context.buffer(vertex_cluster_ids)
    cluster_id_buffer.bind_to_storage_buffer(binding=2)

    #print(first_pass_comp_shader["inVerts"].size)
    #exit()

    #print(np.reshape(np.frombuffer(vertex_buffer.read(),dtype="f4"),newshape=(len(vertices),3))[:5])

    image_shape = (size, 14)
    # Output "image" creation
    cluster_quadric_map_int = fp_context.texture(size=image_shape, components=1, dtype="i4")
    cluster_quadric_map_int.bind_to_image(4, read=True, write=True)

    #quadric_map = fp_context.texture(size=(size,4), components=4,dtype="f4") 

    ############## FP Uniforms ###################

    first_pass_comp_shader['resolution'].value = resolution
    first_pass_comp_shader['float_to_int_scaling_factor'].value  = float_to_int_scaling_factor
    first_pass_comp_shader['debug'].value  = False
    ##############################################

    debug = True
    st = time.time()
    first_pass_comp_shader.run(size, 1, 1)
    print("Running FP Compute Shader Took {:.5f} s".format(time.time()-st))

    if debug:
        print("Cluster Quadric Map FP Output")
        print("\t",np.frombuffer(cluster_quadric_map_int.read(),dtype=np.int32))

    # Output "image" reading
    fp_output_data_original = np.frombuffer(cluster_quadric_map_int.read(),dtype=np.int32)
    #print(output_data[:28])
    fp_output_data = np.reshape(fp_output_data_original, newshape=image_shape, order="F") / float_to_int_scaling_factor
    if debug:
        print("------ First Pass Output------")
        print("Cluster Quadric Map:")
        print(fp_output_data[:,:4])
        print("Vertices Counted Total:",sum(fp_output_data[:,3]))
        print("Actual num of tris:", len(indices))


    fp_output_array = np.array(fp_output_data, dtype=np.float32,order="F")
    '''
    iter, span = 0, 20
    while True:
        print(iter,"to",iter+span, ":", output_array[iter:iter+span])
        iter+=span
        if input("Enter nothing to continue:") != "":
            break'''
    if debug:
        print(fp_output_array)
        output_sum_vertices = fp_output_array[:,:3]
        print(output_sum_vertices.shape)
        output_count_vertices = fp_output_array[:,3]
        
        avg_vertices = np.empty(shape=(1,3),dtype=np.float32,order="F")
        for i in range(len(output_sum_vertices)):
            if output_count_vertices[i] > 0.1:
                avg_vertices = np.concatenate((avg_vertices, np.ndarray(shape=(1,3),
                                    buffer=np.array([output_sum_vertices[i][j]/output_count_vertices[i] for j in range(3)]),
                                    dtype=np.float32,order="F")))

        avg_vertices = avg_vertices[1:]
        print("Avg. Vertices from FP:", avg_vertices.shape)
        print("min,max x:",min(avg_vertices[:,0]), max(avg_vertices[:,0]))
        print("min,max y:",min(avg_vertices[:,1]), max(avg_vertices[:,1]))
        print("min,max z:",min(avg_vertices[:,2]), max(avg_vertices[:,2]))
        print(avg_vertices[:resolution*4,:3])

    print("End of First Pass")

    # End of First Pass

    ##########################################

    # Second Pass
    sp_context = moderngl.create_standalone_context(require=430)
    second_pass_comp_shader = sp_context.compute_shader(source("shaders/secondpass.comp", shader_constants))
    print("Successfully compiled 2nd-pass compute shader!")


    sp_cluster_quadric_map = sp_context.texture(size=image_shape,components=1, 
                                                data=fp_output_data_original, dtype="i4")
    if debug:
        print("Comparing SP Input Cluster Quadric Map to FP Output Quadric Map:")
        print("\t",np.frombuffer(sp_cluster_quadric_map.read(),dtype=np.int32)[:20])
        print("\t",fp_output_data_original[:20])

    sp_cluster_quadric_map.bind_to_image(0, read=True, write=False)

    sp_cluster_vertex_positions = sp_context.texture(size=(image_shape[0],1),components=4, 
                                                data=None, dtype="f4")
    sp_cluster_vertex_positions.bind_to_image(1, read=False, write=True)


    # If commented out, Causes "nan" or "inf" issues
    second_pass_comp_shader['float_to_int_scaling_factor'] = float_to_int_scaling_factor
    #second_pass_comp_shader['resolution'] = resolution

    st = time.time()
    second_pass_comp_shader.run(size, 1, 1)
    print("Running SP Compute Shader Took {:.5f} s".format(time.time()-st))

    #print(sp_output_vertex_positions_original)


    # Second Pass Output "image" reading
    sp_output_vertex_positions_original = np.frombuffer(sp_cluster_vertex_positions.read(),dtype=np.float32)
    sp_output_vertex_positions = np.reshape(sp_output_vertex_positions_original, newshape=(image_shape[0],4), order="C")

    if debug:
        debug_vertex_positions = sp_output_vertex_positions[sp_output_vertex_positions[:,3] > -1.0]
        print(debug_vertex_positions[:,:])
        print("Second Pass cluster vertex positions:", sp_output_vertex_positions.shape,"-->",debug_vertex_positions.shape)
        print("Without empty cells:", sp_output_vertex_positions.shape,"-->",debug_vertex_positions.shape)
        print("\tNon-empty cells:", str(round(debug_vertex_positions.shape[0]/sp_output_vertex_positions.shape[0]*100,2))+"%")
        print("Min,max of x:",min(debug_vertex_positions[:,0]),",",max(debug_vertex_positions[:,0]))
        print("Min,max of y:",min(debug_vertex_positions[:,1]),",",max(debug_vertex_positions[:,1]))
        print("Min,max of z:",min(debug_vertex_positions[:,2]),",",max(debug_vertex_positions[:,2]))
        print(debug_vertex_positions[:resolution*4,:3])
    sp_simplified_vertex_positions_only = sp_output_vertex_positions[:,:3]

    # End of Second Pass

    ##########################################

    # Third Pass
    tp_context = moderngl.create_standalone_context(require=430)
    third_pass_comp_shader = tp_context.compute_shader(source("shaders/thirdpass.comp", shader_constants))
    print("Successfully compiled 3rd-pass compute shader!")


    tp_simplified_vertex_positions = tp_context.texture(
                            size=(size,1), components=4, 
                            data=sp_output_vertex_positions_original,
                            dtype="f4")
    tp_simplified_vertex_positions.bind_to_image(0, read=True, write=False)


    tp_output_indices_texture = tp_context.texture(size=(len(indices),1), components=4, dtype="i4")
    tp_output_indices_texture.bind_to_image(1, read=False, write=True)

    tp_output_tris_texture = tp_context.texture(size=(len(indices),3), components=4, dtype="f4")
    tp_output_tris_texture.bind_to_image(2, read=False, write=True)

    # Create/bind vertex/index data
    tp_vertex_buffer = tp_context.buffer(vertices.astype("f4").tobytes())
    tp_vertex_buffer.bind_to_storage_buffer(binding=0)
    tp_index_buffer = tp_context.buffer(indices.astype("i4").tobytes())
    tp_index_buffer.bind_to_storage_buffer(binding=1)

    tp_cluster_id_buffer = tp_context.buffer(vertex_cluster_ids)
    tp_cluster_id_buffer.bind_to_storage_buffer(binding=2)


    third_pass_comp_shader['resolution'] = resolution

    # Actually run the shader
    st = time.time()
    third_pass_comp_shader.run(len(indices),1,1)
    print("Running 3rd-pass Compute Shader Took {:.5f} s".format(time.time()-st))

    # Second Pass Output "image" reading
    tp_output_vertex_indicies_original = np.frombuffer(tp_output_indices_texture.read(),dtype=np.int32)
    tp_output_vertex_indicies = np.reshape(tp_output_vertex_indicies_original, newshape=(len(indices),4), order="C")

    if True:
        print("3rd-pass output")
        print(tp_output_vertex_indicies.shape)
        print("Number of output tris:",end=" ")
        print(tp_output_vertex_indicies[tp_output_vertex_indicies[:,3] > 0].shape[0])
        print()

    tp_output_vertex_indicies_only = tp_output_vertex_indicies[tp_output_vertex_indicies[:,3] > 0][:,:3] # np.array(tp_output_vertex_indicies[:,:3], dtype=np.int32)
    print(tp_output_vertex_indicies_only)

    tp_output_tris_original = np.frombuffer(tp_output_tris_texture.read(),dtype=np.float32)
    tp_output_tris = np.reshape(tp_output_tris_original, newshape=(len(indices),3,4), order="C")

    if True:
        print("3rd-pass output tris")
        print(tp_output_tris.shape)
        print("Number of output tris:",end=" ")
        print(len(tp_output_tris[tp_output_tris[:, 0, 0] > 0]))
        print()

    tp_output_tris_only = tp_output_tris[tp_output_tris[:,0,0] > 0][:,:3] # np.array(tp_output_vertex_indicies[:,:3], dtype=np.int32)

    print(tp_output_tris_only)

    #exit()

    ### End of Third Pass

''' 
TODO: 
    1. Add shading to model in shaders (get normals in geometry shader)
    2. Make updates in real-time (resolution is user-controlled via keys)
    3. Fix Line Looping issue
    4. More meshes to choose from 
        a. In real time via keys?
    ...?. Look into the issue with resolution > 25 (maybe  1D array memory size limitations?)

'''

class RenderWindow(BasicWindow):
    gl_version = (4, 3)
    title = "Vertex Cluster Quadric Error Metric Mesh Decimation"

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.back_color = (0.3, 0.5, 0.8, 1) #(1,1,1, 1)


        self.prog = self.ctx.program(
            vertex_shader=open("shaders/basic.vert","r").read(),
            #geometry_shader=open("shaders/basic.geom","r").read(),
            fragment_shader=open("shaders/shader.frag","r").read()
            )
        #self.prog["width"].value = self.wnd.width
        #self.prog["height"].value = self.wnd.height
        
        self.prog["bbox.min"] = bbox[0]
        self.prog["bbox.max"] = bbox[1]
        self.prog['model'].value = tuple(transf.compose_matrix(angles=(np.pi/4, np.pi/4, 0)).ravel())
        self.prog['view'].value = tuple(transf.identity_matrix().ravel())
        self.prog['proj'].value = tuple(transf.identity_matrix().ravel()) #tuple(transf.projection_matrix(point=(0,0,0), normal=(0,0,1),direction=(0,0,1), perspective=(0,0,1)).ravel())

        if not renderonly:
            print(sp_simplified_vertex_positions_only.shape)
            #print(sp_simplified_vertex_positions_only[sp_simplified_vertex_positions_only[:,2]>0])
            self.indices = self.ctx.buffer(tp_output_vertex_indicies_only.copy(order="C"))            
            self.vbo = self.ctx.buffer(sp_simplified_vertex_positions_only.copy(order="C"))
            #print(tp_output_vertex_indicies_only[tp_output_vertex_indicies_only[:,0]>0][0,0])
            #print(sp_simplified_vertex_positions_only[1643])
        else:
            self.indices = self.ctx.buffer(indices[:,:3].copy(order="C"))
            self.vbo = self.ctx.buffer(vertices[:,:3].copy(order="C"))
        



        
        #print(self.vbo)
        self.vao = self.ctx.vertex_array(
            self.prog, 
            [
                (self.vbo, '3f', 'inVert'),
            ],
            self.indices
        )

        



    def key_event(self, key, action, modifiers):
        
        # Key presses
        if action == self.wnd.keys.ACTION_PRESS:
            '''
            if key == self.wnd.keys.G and not modifiers.shift and not modifiers.ctrl:
                if self.prog['sphere.glossiness'].value < 1 :
                    self.prog['sphere.glossiness'].value += 0.1
                print("Sphere glossiness:", self.prog['sphere.glossiness'].value)
            '''


        # Key releases
        elif action == self.wnd.keys.ACTION_RELEASE:
            pass
        
    def render(self, run_time, frame_time):
        bc = self.back_color
        self.ctx.front_face = 'ccw'
        self.ctx.clear(bc[0],bc[1],bc[2],bc[3],)
        self.prog["in_color"].value = (0.0, 0.3, 0.8, 1.0)
        self.vao.render(mode=moderngl.TRIANGLES)
        self.prog["in_color"].value = (0.7, 0.2, 0.3, 1.0)
        self.vao.render(mode=moderngl.LINE_LOOP)
        self.prog['model'].value = tuple(transf.compose_matrix(scale=(0.7, 0.7, 0.7),angles=(0, run_time * np.pi/4 ,  0 )).ravel())



if __name__ == '__main__':
    RenderWindow.run()
    #fp_window.run()
