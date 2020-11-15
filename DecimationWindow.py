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

"""
class DecimationProgram(object):

    def __init__(self, vertices, indices, bounding_box=None, resolution_in=20, ):
        self.vertices = vertices
        self.indices = indices
        
        if bounding_box is not None:
            self.bbox = bounding_box
        else:
            self.bbox = utility.bounding_box(self.vertices[:,:3]) # Makes bounding box
       
        self.decimation_context = moderngl.create_standalone_context(require=430)
        self.shader_constants = {
            "NUM_TRIS" : len(self.indices), 
            "X": 1,
            "Y": 1, 
            "Z": 1
        }
        self.resolution = resolution_in
        self.num_clusters = self.resolution**3
        self.float_to_int_scaling_factor = 2**13
        self.image_shape = (self.num_clusters, 14)
        
        _, self.vertex_cluster_ids = get_vertex_cell_indices_ids(vertices=vertices[:,:3], resolution=self.resolution)
        self.vertex_cluster_ids = np.array(self.vertex_cluster_ids, dtype=np.int32)



        self.compute_prog1 = self.decimation_context.compute_shader(source("shaders/firstpass.comp", self.shader_constants))
        self.compute_prog2 = self.decimation_context.compute_shader(source("shaders/secondpass.comp", self.shader_constants))
        self.compute_prog3 = self.decimation_context.compute_shader(source("shaders/thirdpass.comp", self.shader_constants))
        print("\tCompiled all 3 compute shaders!")

        self.vertex_buffer = self.decimation_context.buffer(self.vertices.astype("f4").tobytes())
        self.vertex_buffer.bind_to_storage_buffer(binding=0)
        self.index_buffer = self.decimation_context.buffer(self.indices.astype("i4").tobytes())
        self.index_buffer.bind_to_storage_buffer(binding=1)

        self.cluster_id_buffer = self.decimation_context.buffer(self.vertex_cluster_ids)
        self.cluster_id_buffer.bind_to_storage_buffer(binding=2)

        self.cluster_quadric_map_int = self.decimation_context.texture(size=self.image_shape, components=1, dtype="i4")
        self.cluster_quadric_map_int.bind_to_image(3, read=True, write=True)

        self.cluster_vertex_positions = self.decimation_context.texture(size=(self.num_clusters,1), components=4, 
                                                    data=None, dtype="f4")
        self.cluster_vertex_positions.bind_to_image(4, read=False, write=True)

        self.output_indices = self.decimation_context.texture(size=(len(indices),1), components=4, dtype="i4")
        self.output_indices.bind_to_image(5, read=False, write=True)
        print("Finished Decimation Program Setup!")


    def reset_resolution(self, resolution_in=20):
        self.resolution = resolution_in
        self.num_clusters = self.resolution**3

        self.image_shape = (self.num_clusters, 14)

        # No need to fecompile shaders
        
        # Resize textures
        self.cluster_quadric_map_int.release()
        self.cluster_quadric_map_int = self.decimation_context.texture(size=self.image_shape, components=1, dtype="i4")
        self.cluster_quadric_map_int.bind_to_image(3, read=True, write=True)

        self.cluster_vertex_positions.release()
        self.cluster_vertex_positions = self.decimation_context.texture(size=(self.num_clusters,1), components=4, 
                                                    data=None, dtype="f4")
        self.cluster_vertex_positions.bind_to_image(4, read=False, write=True)

    def increment_resolution(self):
        if self.resolution < 25:
            self.reset_resolution(self.resolution + 1)
    
    def decrement_resolution(self):
        if self.resolution > 2:
            self.reset_resolution(self.resolution - 1)

    def reset_mesh(self, vertices, indices):
        '''
        Resize all textures and recompile shaders 1 and 3
        Optional/For later
        '''
        pass

    def decimate_mesh(self, resolution=25):
        if self.resolution != resolution:
            self.reset_resolution(resolution)


        # Set uniforms
        self.compute_prog1['resolution'].value = self.resolution
        self.compute_prog1['float_to_int_scaling_factor'].value  =  self.float_to_int_scaling_factor
        self.compute_prog1['debug'].value  = False
        self.compute_prog2['float_to_int_scaling_factor'].value  =  self.float_to_int_scaling_factor
        self.compute_prog3['resolution'] = resolution

        # Run programs
        self.compute_prog1.run(self.num_clusters, 1, 1)
        self.compute_prog2.run(self.num_clusters, 1, 1)
        self.compute_prog3.run(len(self.indices), 1, 1)

        # Need the simplified positions and the indices into those vertices
        return self.cluster_vertex_positions, self.output_indices

"""


"""
renderonly = False  

debug = False
resolution = 20

float_to_int_scaling_factor = 2**13
use_box = False
use_triangle = False
"""

def load_model(model="link"):
    
    if model == "link":
        st = time.time()
        #self.obj_mesh = trimesh.exchange.load.load("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj")
        obj_mesh = pywavefront.Wavefront("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj", collect_faces=True)
        print("Loading mesh took {:.2f} s".format(time.time()-st))
        #vertices = np.array(obj_mesh.vertices, dtype='f4')

        # indices are a list of groups of 3 indices into the vertex array, each group representing 1 triangle
        indices = np.array(obj_mesh.mesh_list[0].faces,dtype=np.int32)
        # Using the positive transformed version for ease of distinguishing valid points
        vertices = np.array(utility.positive_vertices(obj_mesh.vertices),dtype="f4")

    elif model == "amphora":
        st = time.time()
        obj_mesh = pywavefront.Wavefront("meshes/meshes_for_graphics/amphora_with_handles_v2_hi_poly.obj", collect_faces=True)
        print("Loading mesh took {:.2f} s".format(time.time()-st))

        indices = np.array(obj_mesh.mesh_list[0].faces,dtype=np.int32)
        vertices = np.array(utility.positive_vertices(obj_mesh.vertices),dtype="f4")

    elif model == "box" :
        box_model = box.Box()
        vertices = np.array(utility.positive_vertices(box_model.vertices), dtype=np.float32)
        indices = np.array(box_model.indicies, dtype=np.int32)
        print("Vertex Count:",len(vertices),"  Tri Count:",len(indices))


    # Solves an issue with the compute shader not getting the indices correct?
    indices = np.append(indices, np.zeros(shape=(len(indices),1),dtype=np.int32), axis=1)
    vertices = np.append(vertices, np.zeros(shape=(len(vertices),1),dtype=np.float32), axis=1)
    return vertices, indices

"""
if False:
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
        "NUM_TRIS" : len(indices), 
        "X": 1,
        "Y": 1, 
        "Z": 1
    }

    #num_clusters = len(indices)
    num_clusters = resolution**3


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

    image_shape = (num_clusters, 14)
    # Output "image" creation
    cluster_quadric_map_int = fp_context.texture(size=image_shape, components=1, dtype="i4")
    cluster_quadric_map_int.bind_to_image(3, read=True, write=True)

    #quadric_map = fp_context.texture(size=(num_clusters,4), components=4,dtype="f4") 

    ############## FP Uniforms ###################

    first_pass_comp_shader['resolution'].value = resolution
    first_pass_comp_shader['float_to_int_scaling_factor'].value  = float_to_int_scaling_factor
    first_pass_comp_shader['debug'].value  = False
    ##############################################

    debug = True
    st = time.time()
    first_pass_comp_shader.run(num_clusters, 1, 1)
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
    
    iter, span = 0, 20
    while True:
        print(iter,"to",iter+span, ":", output_array[iter:iter+span])
        iter+=span
        if input("Enter nothing to continue:") != "":
            break
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

    sp_cluster_quadric_map.bind_to_image(3, read=True, write=False)

    sp_cluster_vertex_positions = sp_context.texture(size=(image_shape[0],1),components=4, 
                                                data=None, dtype="f4")
    sp_cluster_vertex_positions.bind_to_image(4, read=False, write=True)


    # If commented out, Causes "nan" or "inf" issues
    second_pass_comp_shader['float_to_int_scaling_factor'] = float_to_int_scaling_factor
    #second_pass_comp_shader['resolution'] = resolution

    st = time.time()
    second_pass_comp_shader.run(num_clusters, 1, 1)
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
                            size=(num_clusters,1), components=4, 
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

"""

''' 
TODO: 
    X 1. Add shading to model in shaders (get normals in geometry shader)
    2. Make updates in real-time (resolution is user-controlled via keys)
    3. Fix Line Looping issue
    4. More meshes to choose from 
        a. In real time via keys?
    5. Make timer more precise (time.time_ns())
        (and perform average execution time for real-time calcs???)
    6. Get actual output tri/vert count 
        (basically take the output buffers and get # of non-errored lines)
    ...?. Look into the issue with resolution > 25 (maybe  1D array memory size limitations?)

'''


class DecimationWindow(BasicWindow):
    gl_version = (4, 3)
    title = "Vertex Cluster Quadric Error Metric Mesh Decimation"

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.back_color = (0.3, 0.5, 0.8, 1) #(1,1,1, 1)
        self.is_decimated = False
        self.debug = False
        self.indexed_output = True

        self.vertices, self.indices = load_model("amphora")
        print(self.vertices.shape)
        self.bbox = utility.bounding_box(points=self.vertices[:,:3])

        self.tri_prog = self.ctx.program(
            vertex_shader=open("shaders/basic.vert","r").read(),
            geometry_shader=open("shaders/basic.geom","r").read(),
            fragment_shader=open("shaders/shader.frag","r").read()
            )
        self.line_prog = self.ctx.program(
            vertex_shader=open("shaders/basic.vert","r").read(),
            fragment_shader=open("shaders/basic.frag","r").read()
            )
        #self.prog["width"].value = self.wnd.width
        #self.prog["height"].value = self.wnd.height
        
        self.tri_prog["bbox.min"] = self.bbox[0]
        self.tri_prog["bbox.max"] = self.bbox[1]
        self.tri_prog['model'].value = tuple(transf.compose_matrix(angles=(0, np.pi*5/4, 0)).ravel())# (np.pi/4, np.pi/4, 0)).ravel())
        self.tri_prog['view'].value = tuple(transf.identity_matrix().ravel())
        self.tri_prog['proj'].value = tuple(transf.identity_matrix().ravel()) 
        self.tri_prog["in_color"].value = (0.0, 0.9, 0.3, 1.0)

        self.line_prog["bbox.min"] = self.bbox[0]
        self.line_prog["bbox.max"] = self.bbox[1]
        self.line_prog['model'].value = tuple(transf.compose_matrix(angles=(np.pi/4, np.pi/4, 0)).ravel())
        self.line_prog['view'].value = tuple(transf.identity_matrix().ravel())
        self.line_prog['proj'].value = tuple(transf.identity_matrix().ravel()) 
        self.line_prog["in_color"].value = (0.7, 0.2, 0.3, 1.0)


        self.vbo = self.ctx.buffer(self.vertices[:,:3].copy(order="C"))
        self.index_buffer = self.ctx.buffer(self.indices[:,:3].copy(order="C"))
        

        #print(self.vbo)
        self.tri_vao_base = self.ctx.vertex_array(
            self.tri_prog, 
            [
                (self.vbo, '3f', 'inVert'),
            ],
            self.index_buffer
        )

        self.line_vao_base = self.ctx.vertex_array(
            self.line_prog, 
            [
                (self.vbo, '3f', 'inVert'),
            ],
            self.index_buffer
        )

        self.current_tri_vao = self.tri_vao_base
        self.current_line_vao = self.line_vao_base

        self.shader_constants = {
            "NUM_TRIS" : len(self.indices), 
            "X": 1,
            "Y": 1, 
            "Z": 1
        }
        self.resolution = 25
        self.num_clusters = self.resolution**3
        self.float_to_int_scaling_factor = 2**13
        self.image_shape = (self.num_clusters, 14)
        
        _, self.vertex_cluster_ids = get_vertex_cell_indices_ids(vertices=self.vertices[:,:3], resolution=self.resolution)
        self.vertex_cluster_ids = np.array(self.vertex_cluster_ids, dtype=np.int32)


        self.compute_prog1 = self.ctx.compute_shader(source("shaders/firstpass.comp", self.shader_constants))
        self.compute_prog2 = self.ctx.compute_shader(source("shaders/secondpass.comp", self.shader_constants))
        self.compute_prog3 = self.ctx.compute_shader(source("shaders/thirdpass.comp", self.shader_constants))
        print("\tCompiled all 3 compute shaders!")

        self.vertex_buffer = self.ctx.buffer(self.vertices.astype("f4").tobytes())
        self.vertex_buffer.bind_to_storage_buffer(binding=0)
        self.index_buffer = self.ctx.buffer(self.indices.astype("i4").tobytes())
        self.index_buffer.bind_to_storage_buffer(binding=1)

        self.cluster_id_buffer = self.ctx.buffer(self.vertex_cluster_ids)
        self.cluster_id_buffer.bind_to_storage_buffer(binding=2)

        self.cluster_quadric_map_int = self.ctx.texture(size=self.image_shape, components=1, dtype="i4")
        self.cluster_quadric_map_int.bind_to_image(3, read=True, write=True)

        self.cluster_vertex_positions = self.ctx.texture(size=(self.num_clusters,1), components=4, 
                                                    data=None, dtype="f4")
        self.cluster_vertex_positions.bind_to_image(4, read=False, write=True)

        self.output_indices = self.ctx.texture(size=(len(self.indices),1), components=4, dtype="i4")
        self.output_indices.bind_to_image(5, read=False, write=True)

        self.output_tri_verts = self.ctx.texture(size=(self.num_clusters,1), components=4, dtype="f4")
        self.output_tri_verts.bind_to_image(6, read=False, write=True)

        print("Finished Decimation Program Setup!")

        self.decimate_mesh()
        # VAOs set in decimate_mesh()
        if self.debug:
            self.debug_dump()



    def reset_resolution(self, resolution_in):
        self.resolution = resolution_in
        self.num_clusters = self.resolution**3

        self.image_shape = (self.num_clusters, 14)
        
        _, self.vertex_cluster_ids = get_vertex_cell_indices_ids(vertices=self.vertices[:,:3], resolution=self.resolution)
        self.vertex_cluster_ids = np.array(self.vertex_cluster_ids, dtype=np.int32)
        
        # Have data, now have to refill the buffer with it
        #self.cluster_id_buffer.release()
        self.cluster_id_buffer = self.ctx.buffer(self.vertex_cluster_ids)
        self.cluster_id_buffer.bind_to_storage_buffer(binding=2)

        #print(max(self.vertex_cluster_ids))

        # No need to recompile shaders
        
        # Resize textures
        #self.cluster_quadric_map_int.release()
        self.cluster_quadric_map_int = self.ctx.texture(size=self.image_shape, data=None, components=1, dtype="i4")
        self.cluster_quadric_map_int.bind_to_image(3, read=True, write=True)

        #self.cluster_vertex_positions.release()
        self.cluster_vertex_positions = self.ctx.texture(size=(self.num_clusters,1), data=None, components=4, dtype="f4")
        self.cluster_vertex_positions.bind_to_image(4, read=False, write=True)

        #self.output_tri_verts.release()
        self.output_tri_verts = self.ctx.texture(size=(self.num_clusters,1), data=None, components=4, dtype="f4")
        self.output_tri_verts.bind_to_image(6, read=False, write=True)


    def increment_resolution(self):
        if self.resolution < 25:
            self.reset_resolution(self.resolution + 1)
            self.decimate_mesh()
            return True
        else:
            print("Already at Max Resolution!")
            return False

    def decrement_resolution(self):
        if self.resolution > 2:
            self.reset_resolution(self.resolution - 1)            
            self.decimate_mesh()
            return True
        else:
            print("Already at Min Resolution!")
            return False

    def reset_mesh(self, vertices, indices):
        '''
        Resize all textures and recompile shaders 1 and 3
        Optional/For later
        '''
        pass

    def decimate_mesh(self):
        print("Current resolution:", self.resolution)
        #print(self.indices)
        
        # Set uniforms
        self.compute_prog1['resolution'].value = self.resolution
        self.compute_prog1['float_to_int_scaling_factor'].value  =  self.float_to_int_scaling_factor
        self.compute_prog1['debug'].value  = False

        self.compute_prog2['float_to_int_scaling_factor'].value  =  self.float_to_int_scaling_factor

        self.compute_prog3['resolution'] = self.resolution

        # Run programs
        self.compute_prog1.run(self.num_clusters, 1, 1)
        print(np.frombuffer(self.cluster_vertex_positions.read(),dtype=np.float32)) # These print statements, for some reason, seem to be really important for timing reasons
        self.compute_prog2.run(self.num_clusters, 1, 1)
        print(np.frombuffer(self.cluster_vertex_positions.read(),dtype=np.float32)) # These print statements, for some reason, seem to be really important for timing reasons
        self.compute_prog3.run(len(self.indices), 1, 1)

        # Need the simplified positions and the indices into those vertices
        self.output_vertices_array = np.reshape(np.frombuffer(self.cluster_vertex_positions.read(),dtype=np.float32),
                                    newshape=(self.num_clusters,4), order="C")
        self.dec_vert_buff = self.ctx.buffer(self.output_vertices_array[:,:3].copy(order="C"))

        self.output_indices_array = np.reshape(np.frombuffer(self.output_indices.read(),dtype=np.int32),
                                         newshape=(len(self.indices),4), order="C")
        self.output_indices_array = self.output_indices_array[self.output_indices_array[:,3] > 0]
        self.dec_index_buff = self.ctx.buffer(self.output_indices_array[:,:3].copy(order="C"))

        self.output_tri_verts_array = np.reshape(np.frombuffer(self.output_tri_verts.read(),dtype=np.float32),
                                         newshape=(self.num_clusters,4), order="C")
        self.output_tri_verts_array = self.output_tri_verts_array[self.output_tri_verts_array[:,3] > 0]
        self.dec_tri_vert_buff = self.ctx.buffer(self.output_tri_verts_array[:,:3].copy(order="C"))
        

        self.tri_vao_decimated = self.ctx.vertex_array(
            self.tri_prog, 
            [
                (self.dec_vert_buff, '3f', 'inVert'),
            ],
            self.dec_index_buff
        )

        self.line_vao_decimated = self.ctx.vertex_array(
            self.line_prog, 
            [
                (self.dec_vert_buff, '3f', 'inVert'),
            ],
            self.dec_index_buff
        )

        self.tri_only_vao_decimated = self.ctx.vertex_array(
            self.tri_prog, 
            [
                (self.dec_tri_vert_buff, '3f', 'inVert'),
            ],
        )

        self.tri_only_line_vao_decimated = self.ctx.vertex_array(
            self.line_prog, 
            [
                (self.dec_tri_vert_buff, '3f', 'inVert'),
            ],
        )


    def set_vertex_array_object(self):
        if self.is_decimated:
            if self.indexed_output:
                self.current_tri_vao = self.tri_vao_decimated
                self.current_line_vao = self.line_vao_decimated
            else:
                self.current_tri_vao = self.tri_only_vao_decimated
                self.current_line_vao = self.tri_only_line_vao_decimated
        else:                    
            self.current_tri_vao = self.tri_vao_base
            self.current_line_vao = self.line_vao_base


    def debug_dump(self):
        print("-------DEBUG VALUES------")
        print("Resolution=",self.resolution)
        print("First Pass:")
        cluster_map_output = np.frombuffer(self.cluster_quadric_map_int.read(), dtype=np.int32) / self.float_to_int_scaling_factor
        cluster_map_output = np.reshape(np.array(cluster_map_output,dtype=np.float32,order="F"), newshape=(self.num_clusters, 14),order="F")
        print("    Cluster map output:", len(cluster_map_output), "clusters...")
        #print(cluster_map_output)
        
        avg_vertices = np.empty(shape=(1,3),dtype=np.float32,order="F")
        for i in range(len(cluster_map_output)):
            if cluster_map_output[i,3] > 0.1:
                avg_vertices = np.concatenate((avg_vertices, np.ndarray(shape=(1,3),
                                    buffer=np.array([cluster_map_output[i,j]/cluster_map_output[i,3] for j in range(3)]),
                                    dtype=np.float32,order="F")))
        avg_vertices = avg_vertices[1:]

        print("    Valid Clusters:")

        print("\t",str(avg_vertices).replace("\n","\n\t").replace("[[","[").replace("]]","]"))
        
        print("Second Pass:")
        print(np.frombuffer(self.cluster_vertex_positions.read(), dtype=np.float32) )
        print("    Valid Optimal Vertex Output:")

        sp_avg_vertices = np.frombuffer(self.cluster_vertex_positions.read(), dtype=np.float32) 
        sp_avg_vertices = np.reshape(np.array(sp_avg_vertices,dtype=np.float32,order="C"), newshape=(self.num_clusters, 4), order="C")
        sp_avg_vertices = sp_avg_vertices[sp_avg_vertices[:,3] > 0]
        print("\t", str(sp_avg_vertices).replace("\n","\n\t").replace("[[","[").replace("]]","]"))



        #debug_output_indices = np.reshape(np.frombuffer(self.dec_index_buff.read(),dtype=np.int32),
        #        newshape=(len(self.indices), 3))
        '''print("Output Indices: Num=",len(self.output_indices_array))
        ind = self.output_indices_array
        print("Bad indices:",len(ind[ind[:,0] < -0.5]))
        print(self.output_indices_array[:5])
        #debug_output_vertices_array = np.reshape(np.frombuffer(self.dec_vert_buff.read(),dtype=np.float32),
         #       newshape=(self.num_clusters, 3))
        print(self.output_vertices_array[:20])
        valid_verts = self.output_vertices_array[self.output_vertices_array[:,0] > -0.5]
        print("Output Vertices: Num=",len(valid_verts))
        print(valid_verts[:20])

        print("Chosen Vertices:",end="")
        included_indices = np.unique(self.output_indices_array.flatten())
        vertices = self.output_vertices_array[included_indices]
        print(len(vertices[vertices[:,0] > -0.5]))
        print("Accidentally included verts:",end="")
        print(len(vertices[vertices[:,0] < -0.5]))
        print(vertices[vertices[:,0] < -0.5])
        '''



    def key_event(self, key, action, modifiers):
        
        # Key presses
        if action == self.wnd.keys.ACTION_PRESS:

            # Toggle showing decimation
            if key == self.wnd.keys.D:
                if modifiers.shift:
                    self.indexed_output = not self.indexed_output
                else:
                    self.is_decimated = not self.is_decimated
                    self.set_vertex_array_object()
                    print("Showing Decimated Mesh:",self.is_decimated)

            if key == self.wnd.keys.R and not modifiers.ctrl:
                result = False
                if not modifiers.shift:
                    result = self.increment_resolution()
                else:
                    result = self.decrement_resolution()

                if result:
                    if self.debug:
                        self.debug_dump()
                    self.set_vertex_array_object()


            '''
                if self.prog['sphere.glossiness'].value < 1 :
                    self.prog['sphere.glossiness'].value += 0.1
                print("Sphere glossiness:", self.prog['sphere.glossiness'].value)
            '''


        # Key releases
        elif action == self.wnd.keys.ACTION_RELEASE:
            pass
        
    def render(self, run_time, frame_time):
        bc = self.back_color
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.front_face = 'cw'
        self.ctx.clear(bc[0],bc[1],bc[2],bc[3],)
        self.current_tri_vao.render(mode=moderngl.TRIANGLES)
        self.current_line_vao.render(mode=moderngl.LINE_LOOP)
        self.tri_prog['model'].value = tuple(transf.compose_matrix(scale=(0.7, 0.7, 0.7),angles=(0, run_time * np.pi/4 ,  0 )).ravel())
        self.line_prog['model'].value = tuple(transf.compose_matrix(scale=(0.7, 0.7, 0.7),angles=(0, run_time * np.pi/4 ,  0 )).ravel())



if __name__ == '__main__':
    DecimationWindow.run()
    #fp_window.run()
