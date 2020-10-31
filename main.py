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


class RayMarchingWindow(BasicWindow):
    gl_version = (4, 3)
    title = "Geometry Shader Testing"

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader= open("shader.vert", "r").read(),
            geometry_shader=open("miniTris.geom", "r").read(),
            fragment_shader='''
                #version 330
                out vec4 f_color;
                void main() {
                    f_color = vec4(0.3, 0.5, 1.0, 1.0);
                }
            ''',
        )
        #self.vao = self.ctx.vertex_array(self.prog, [])

        
        self.back_color = (0, 0.3, 0.9, 1.0) #(1,1,1, 1)
        st = time.time()
        self.obj_mesh = trimesh.load("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj", file_type='obj', force="mesh")
        print("Loading took {:.2f} s".format(time.time()-st))

        vertices = np.array(self.obj_mesh.vertices, dtype='f4')
        bbox = utility.bounding_box(vertices) # Makes bounding box
        self.prog['bbox.min'].value = bbox[0]
        self.prog['bbox.max'].value = bbox[1]


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

        '''
        # Initialize an empty 3D texture for octree
        self.cell_texture = self.ctx.texture(size=, len(vertices), components=4, 
                            data=np.zeros((len(vertices),4),dtype=float, order='C'),
                            alignment=1,
                            dtype="f4")
        '''
        
        self.vbo = self.ctx.buffer(vertices)

        print(self.vbo)
        self.vao = self.ctx.vertex_array(
            self.prog, 
            [
                (self.vbo, '3f', 'inVert'),
            ]
        )
        print("Made it here")
        



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
        

if __name__ == '__main__':
    RayMarchingWindow.run()
    #wind = RayMarchingWindow()
    #wind.render()
