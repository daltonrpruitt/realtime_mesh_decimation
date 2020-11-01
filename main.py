'''
    Dalton Winans-Pruitt  dp987
    Based on Hello World example from https://github.com/moderngl/moderngl/tree/master/examples
    Testing geometry shader usage
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

class FirstPassWindow(BasicWindow):
    gl_version = (4, 3)
    title = "Geometry Shader Testing"

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        
        self.miniTrisProg = self.ctx.program(
            vertex_shader=open("shaders/basic.vert", "r").read(),
            geometry_shader=open("shaders/miniTris.geom", "r").read(),
            fragment_shader=open("shaders/basic.frag", "r").read(),
        )
        #self.vao = self.ctx.vertex_array(self.prog, [])
        '''
        self.basicProg = self.ctx.program(
            vertex_shader=open("basic.vert", "r").read(),
            geometry_shader=open("miniTris.geom", "r").read(),
            fragment_shader=open("basic.frag", "r").read(),
        )
        '''
        self.cluster_quadric_map_generation = self.ctx.program(
            vertex_shader=open("shaders/first_pass/cell_calc.vert", "r").read(),
            geometry_shader=open("shaders/first_pass/quadric_calc.geom", "r").read(),
            fragment_shader=open("shaders/first_pass/render_quadric.frag", "r").read(),

        )
        
        self.mini_tris = False
        self.first_pass = True
        self.first_pass_output = False
        resolution = 10  # Quadric Cell resolutin (# in each dimension)

        self.back_color = (0, 0.3, 0.9, 1.0) #(1,1,1, 1)
        st = time.time()
        #self.obj_mesh = trimesh.exchange.load.load("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj")
        self.obj_mesh = pywavefront.Wavefront("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj", collect_faces=True)
        print("Loading took {:.2f} s".format(time.time()-st))
        vertices = np.array(self.obj_mesh.vertices, dtype='f4')
        bbox = utility.bounding_box(vertices) # Makes bounding box
        #print(len(vertices))
        #self.obj_mesh.show()
        #print(self.obj_mesh.vertices[:10])
        #print(self.obj_mesh.mesh_list[0].faces[:10])
        #exit()

        if self.mini_tris:
            self.miniTrisProg['bbox.min'].value = bbox[0]
            self.miniTrisProg['bbox.max'].value = bbox[1]
            #print(tuple(transf.compose_matrix(angles=(0, np.pi/2, 0)).ravel()))
            self.miniTrisProg['model'].value = tuple(transf.compose_matrix(angles=(0, np.pi/2, 0)).ravel())
            self.miniTrisProg['view'].value = tuple(transf.identity_matrix().ravel())
            self.miniTrisProg['proj'].value = tuple(transf.identity_matrix().ravel())

            indices = np.array(self.obj_mesh.mesh_list[0].faces)

            #print('Vertices 0-9', vertices[:10])
            #print('Indices 0-9', indices[:10])


            '''

            vertices = np.array([
                -0.5, -0.5, 0,
                0.5, -0.5, 0,
                -0.5, 0.5, 0,
                0.5, 0.5, 0,

            ], dtype='f4')

            # https://github.com/moderngl/moderngl/blob/master/examples/raymarching.py
            idx_data = np.array([
                0, 1, 2, 3
            ])
            idx_buffer = self.ctx.buffer(idx_data)
            '''            
            self.vbo = self.ctx.buffer(vertices)
            
            print(self.vbo)
            self.vao = self.ctx.vertex_array(
                self.miniTrisProg, 
                [
                    (self.vbo, '3f', 'inVert'),
                ]
            )
        elif self.first_pass:

            self.ctx.blend_func = self.ctx.ADDITIVE_BLENDING # Required to add quadrics together

            self.cluster_quadric_map_generation['bbox.min'].value = bbox[0]
            self.cluster_quadric_map_generation['bbox.max'].value = bbox[1]

            self.cluster_quadric_map_generation['cell_full_scale'].value = 100
            self.cluster_quadric_map_generation['resolution'].value = resolution
            self.wnd.size = (resolution**2, resolution)
            print(self.wnd.size)
            #self.cluster_quadric_map_generation['width'].value = self.wnd.width
            #self.cluster_quadric_map_generation['height'].value = self.wnd.height


            indices = np.array(self.obj_mesh.mesh_list[0].faces)

            #print('Vertices 0-9', vertices[:10])
            #print('Indices 0-9', indices[:10])

            # Initialize an empty array texture for octree
            self.cell_texture = self.ctx.texture(size=(resolution**2,resolution), components=4, 
                                data=np.zeros(resolution**3 * 4,dtype="f4", order='C'),
                                alignment=1,
                                dtype="f4")
                
            ''' May be used later?
            self.vertex_texture = self.ctx.texture(size=(len(vertices),1), components=4, 
                                data=np.zeros((len(vertices)*4),dtype="f4", order='C'),
                                alignment=1,
                                dtype="f4")   
            '''                  
            print("cell texture size =",self.cell_texture._size)
            #exit()
            self.cell_framebuffer = self.ctx.framebuffer(color_attachments=[self.cell_texture])
            self.vbo = self.ctx.buffer(vertices)
            
            print(self.vbo)
            self.vao = self.ctx.vertex_array(
                self.cluster_quadric_map_generation, 
                [
                    (self.vbo, '3f', 'inVert'),
                ]
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
        if self.first_pass:
            self.ctx.clear(1.0, 1.0, 1.0)
            #self.vao.render(mode=moderngl.POINTS)
            if not self.first_pass_output:
                self.first_pass_output = True
                with open("first_pass_output.bin","wb") as output_file:
                    output_file.write(self.cell_framebuffer.read())
                self.first_pass = False
        elif not self.first_pass:
            self.close()


        elif self.mini_tris:
            #self.ctx.clear(self.back_color)
            self.ctx.clear(1.0, 1.0, 1.0)
            #self.vao.render(mode=moderngl.POINTS, vertices=100, instances=2)
            self.vao.render(mode=moderngl.POINTS)
            self.miniTrisProg['model'].value = tuple(transf.compose_matrix(angles=(0, np.pi/2 * run_time/8, 0)).ravel())

        

if __name__ == '__main__':
    FirstPassWindow.run()
    #fp_window.run()
