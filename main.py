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
import utility
import transformations as transf

class RayMarchingWindow(BasicWindow):
    gl_version = (4, 3)
    title = "Geometry Shader Testing"

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        '''
        self.miniTrisProg = self.ctx.program(
            vertex_shader=open("basic.vert", "r").read(),
            geometry_shader=open("miniTris.geom", "r").read(),
            fragment_shader=open("basic.frag", "r").read(),
        )'''
        #self.vao = self.ctx.vertex_array(self.prog, [])
        '''
        self.basicProg = self.ctx.program(
            vertex_shader=open("basic.vert", "r").read(),
            geometry_shader=open("miniTris.geom", "r").read(),
            fragment_shader=open("basic.frag", "r").read(),
        )
        '''
        self.cluster_quadric_map_generation = self.ctx.program(
            vertex_shader=open("cell_calc.vert", "r").read(),
            geometry_shader=open("quadric_calc.geom", "r").read(),
            fragment_shader=open("render_quadric.frag", "r").read(),

        )
        
        mini_tris = False
        first_pass = True

        self.back_color = (0, 0.3, 0.9, 1.0) #(1,1,1, 1)
        st = time.time()
        self.obj_mesh = trimesh.load("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj", file_type='obj', force="mesh")
        print("Loading took {:.2f} s".format(time.time()-st))
        vertices = np.array(self.obj_mesh.vertices, dtype='f4')
        bbox = utility.bounding_box(vertices) # Makes bounding box


        if mini_tris:
            self.miniTrisProg['bbox.min'].value = bbox[0]
            self.miniTrisProg['bbox.max'].value = bbox[1]
            #print(tuple(transf.compose_matrix(angles=(0, np.pi/2, 0)).ravel()))
            self.miniTrisProg['model'].value = tuple(transf.compose_matrix(angles=(0, np.pi/2, 0)).ravel())
            self.miniTrisProg['view'].value = tuple(transf.identity_matrix().ravel())
            self.miniTrisProg['proj'].value = tuple(transf.identity_matrix().ravel())

            self.miniTrisProg['cell_full_scale'].value = 100
            self.miniTrisProg['resolution'].value = 10


            indices = np.array(self.obj_mesh.faces)

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

            
            # Initialize an empty array texture for octree
            self.cell_texture = self.ctx.texture(size=len(vertices), components=(4,4), 
                                data=np.zeros((len(vertices)*4),dtype=float, order='C'),
                                alignment=1,
                                dtype="f4")
            print(type(self.cell_texture))
            exit()
            
            self.vbo = self.ctx.buffer(vertices)
            
            print(self.vbo)
            self.vao = self.ctx.vertex_array(
                self.miniTrisProg, 
                [
                    (self.vbo, '3f', 'inVert'),
                ]
            )
        elif first_pass:
            self.cluster_quadric_map_generation['bbox.min'].value = bbox[0]
            self.cluster_quadric_map_generation['bbox.max'].value = bbox[1]
            #print(tuple(transf.compose_matrix(angles=(0, np.pi/2, 0)).ravel()))
            #self.cluster_quadric_map_generation['model'].value = tuple(transf.compose_matrix(angles=(0, np.pi/2, 0)).ravel())
            #self.cluster_quadric_map_generation['view'].value = tuple(transf.identity_matrix().ravel())
            #self.cluster_quadric_map_generation['proj'].value = tuple(transf.identity_matrix().ravel())

            self.cluster_quadric_map_generation['cell_full_scale'].value = 100
            self.cluster_quadric_map_generation['resolution'].value = 10


            indices = np.array(self.obj_mesh.faces)

            #print('Vertices 0-9', vertices[:10])
            #print('Indices 0-9', indices[:10])

            
            # Initialize an empty array texture for octree
            self.cell_texture = self.ctx.texture(size=len(vertices), components=4, 
                                data=np.zeros((len(vertices)*4),dtype=float, order='C'),
                                alignment=1,
                                dtype="f4")
            print(type(self.cell_texture))
            exit()
            
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
        #self.ctx.clear(self.back_color)
        self.ctx.clear(1.0, 1.0, 1.0)
        #self.vao.render(mode=moderngl.POINTS, vertices=100, instances=2)
        self.vao.render(mode=moderngl.POINTS)
        self.miniTrisProg['model'].value = tuple(transf.compose_matrix(angles=(0, np.pi/2 * run_time/8, 0)).ravel())

        

if __name__ == '__main__':
    RayMarchingWindow.run()
    #wind = RayMarchingWindow()
    #wind.render()
