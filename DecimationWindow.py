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

    elif model == "bunny":
        st = time.time()
        obj_mesh = pywavefront.Wavefront("meshes/meshes_for_graphics/bunny_lo_poly.obj", collect_faces=True)
        print("Loading mesh took {:.2f} s".format(time.time()-st))

        indices = np.array(obj_mesh.mesh_list[0].faces,dtype=np.int32)
        vertices = np.array(utility.positive_vertices(obj_mesh.vertices),dtype="f4")
    
    elif model == "teapot":
        st = time.time()
        obj_mesh = pywavefront.Wavefront("meshes/meshes_for_graphics/teapot_ascii_normals_uv.obj", collect_faces=True)
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


''' 
TODO: 
    X 1. Add shading to model in shaders (get normals in geometry shader)
    X 2. Make updates in real-time (resolution is user-controlled via keys)
    3. Fix Line Looping issue
    X 4. More meshes to choose from 
       () a. In real time via keys?
    5. Make timer more precise (time.time_ns())
        (and perform average execution time for real-time calcs???)
    
    X? did percentage
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
        self.use_avg_vertices = False
        self.use_face_area_correction = False
        self.show_lines = False
        self.indexed_output = True
        self.is_animated = False
        self.x_angle = 0
        self.model_matrix = tuple(transf.compose_matrix(scale=(1, 1, 1),angles=(0,  np.pi/1 ,  0 )).ravel())

        self.vertices, self.indices = load_model("link")
        print("Base Mesh: Vertices="+"{:d}".format(len(self.vertices))+" Triangles="+"{:d}".format(len(self.indices)))
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
        self.tri_prog['model'].value = self.model_matrix# (np.pi/4, np.pi/4, 0)).ravel())
        self.tri_prog['view'].value = tuple(transf.identity_matrix().ravel())
        self.tri_prog['proj'].value = tuple(transf.identity_matrix().ravel()) 
        self.tri_prog["in_color"].value = (0.9, 0.9, 0.3, 1.0)

        self.line_prog["bbox.min"] = self.bbox[0]
        self.line_prog["bbox.max"] = self.bbox[1]
        self.line_prog['model'].value = self.model_matrix
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
        self.resolution = 10
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
        self.compute_prog1['face_area_correction'].value = self.use_face_area_correction 


        self.compute_prog2['float_to_int_scaling_factor'].value  =  self.float_to_int_scaling_factor
        self.compute_prog2['use_avg'].value  =  self.use_avg_vertices

        self.compute_prog3['resolution'] = self.resolution

        start_time = time.time()

        # Run programs ... and wait for them to finish completely...
        #print("Indicies Size: ",self.indices.shape[0])
        
        self.compute_prog1.run(self.indices.shape[0], 1, 1)
        self.ctx.finish()
        fp_time = time.time()

        self.compute_prog2.run(self.num_clusters, 1, 1)
        self.ctx.finish()
        sp_time = time.time()

        self.compute_prog3.run(len(self.indices), 1, 1)
        self.ctx.finish()
        tp_time = time.time()

        print("Time to complete passes:")
        print("\tFirst Pass:","{:.8f} ms".format((fp_time - start_time)*10**3))
        print("\tSecond Pass:","{:.8f} ms".format((sp_time - fp_time)*10**3))
        print("\tThird Pass:","{:.8f} ms".format((tp_time - sp_time)*10**3))


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
        
        print("Decimated Mesh: Vertices=" + "{:d}".format(len(self.output_vertices_array[self.output_vertices_array[:,0] > -0.5])) +
                                 " Triangles=" + "{:d}".format(len(self.output_indices_array)))

        print("% decimated: Vertices=" + "{:.1f}".format(100*(1.0-len(self.output_vertices_array[self.output_vertices_array[:,0] > -0.5])/len(self.vertices))) +
                                 " Triangles=" + "{:.1f}".format(100*(1.0-len(self.output_indices_array)/len(self.indices))))
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

        '''
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
        '''


    def set_vertex_array_object(self):
        if self.is_decimated:
            #if self.indexed_output:
            self.current_tri_vao = self.tri_vao_decimated
            self.current_line_vao = self.line_vao_decimated
            # else:
            #     self.current`_tri_vao = self.tri_only_vao_decimated
            #     self.current_line_vao = self.tri_only_line_vao_decimated
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

            if key == self.wnd.keys.A:
                self.use_avg_vertices = not self.use_avg_vertices
                self.decimate_mesh()
                self.set_vertex_array_object()

                print("Using average vertices in 2nd pass:",self.use_avg_vertices)
            
            if key == self.wnd.keys.F:
                self.use_face_area_correction = not self.use_face_area_correction
                self.decimate_mesh()
                self.set_vertex_array_object()

                print("Using face area correction in 1st pass:",self.use_face_area_correction)

            if key == self.wnd.keys.L:
                self.show_lines = not self.show_lines # no print; can see

            if key == self.wnd.keys.Z:
                self.is_animated = not self.is_animated

            if key == self.wnd.keys.X:
                if not modifiers.shift:
                    self.x_angle += np.pi / 12
                else:
                    self.x_angle -= np.pi / 12

        # Key releases
        elif action == self.wnd.keys.ACTION_RELEASE:
            pass
        
    def render(self, run_time, frame_time):
        bc = self.back_color
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.front_face = 'cw' # I do not currently know why everything is clockwise, but ¯\_(ツ)_/¯ 
        self.ctx.clear(bc[0],bc[1],bc[2],bc[3],)
        
        self.current_tri_vao.render(mode=moderngl.TRIANGLES)
        if self.show_lines:
            self.current_line_vao.render(mode=moderngl.LINE_LOOP)
        
        if self.is_animated:
            self.model_matrix = tuple(transf.compose_matrix(scale=(1., 1., 1.),angles=(self.x_angle, run_time * np.pi/4 ,  0 )).ravel())
            self.tri_prog['model'].value = self.model_matrix
            self.line_prog['model'].value = self.model_matrix

if __name__ == '__main__':
    DecimationWindow.run()
    #fp_window.run()
