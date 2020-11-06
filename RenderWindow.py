'''
    Dalton Winans-Pruitt  dp987
    Based on Hello World example from https://github.com/moderngl/moderngl/tree/master/examples
    Basic Window Class for rendering a mesh
'''

import moderngl

from moderngl_window  as mglw
import numpy as np
from window_setup import BasicWindow  # imports moderngl_window

from moderngl import Uniform
import time
import trimesh
import pywavefront
import utility
import transformations as transf
import os


render_ctx = moderngl.create_context(require=430)
render_prog = render_ctx.program(
    vertex_shader="shaders/basic.vert",
    fragment_shader="shaders/basic.frag"
)
render_prog["bbox.min"] = bbox[0]
render_prog["bbox.max"] = bbox[1]

render_vao = render_ctx.vertex_array(
    [
        vertices, "3f4", "inVert"
    ]
)

render_vao.render(moderngl.TRIANGLES)

exit()

''' TODO for First Pass: 
    1. Get vertex and index data to compute shader; and get output buffer recognized
    2. Perform calculations on vertices by using index data
    3. Output the required 14 floats of data (and maybe the extra cluster id) to output buffer
    4. Save the buffer to a file and check its contents to verify it is at least coherent
    5. Cleanup code (if have time)
'''

''' TODO for Second Pass: 
    1. Finish First Pass...
    ...
    ?.. Cleanup 
'''


class RenderWindow(mglw.WindowConfig):
    gl_version = (4, 3)
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4

    title = "Geometry Shader Testing"

    def __init__(self, vertices=None, indices=None, **kwargs ):
        
        super().__init__(**kwargs)

        self.back_color = (1.0, 1.0, 1.0, 1.0) #(1,1,1, 1)


        if vertices is None:
            print("Cannot render 0 vertices!")
            return None

        
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

            self.cluster_quadric_map_generation_prog['bbox.min'].value = bbox[0]
            self.cluster_quadric_map_generation_prog['bbox.max'].value = bbox[1]

            #self.cluster_quadric_map_generation_prog['cell_full_scale'].value = self.resolution
            #self.cluster_quadric_map_generation_prog['resolution'].value = self.resolution
            
            
            self.wnd.size = (self.resolution**2, self.resolution)
            self.wnd.resize(self.resolution**2, self.resolution)
            #print(self.wnd.size)
            #exit()

            #self.cluster_quadric_map_generation_prog['width'].value = self.wnd.width
            #self.cluster_quadric_map_generation_prog['height'].value = self.wnd.height


            index_buffer = self.ctx.buffer(indices)
            #print('Vertices 0-9', vertices[:10])
            #print('Indices 0-9', indices[:10])

            # Initialize an empty array texture for octree
            self.cell_texture = self.ctx.texture(
                                size=(self.resolution**2,self.resolution * 4), 
                                components=4, 
                                data=np.zeros(self.resolution**3 * 4 * 4,dtype=np.float32, order='C'),
                                alignment=1,
                                dtype="f4")
                
            ''' May be used later?
            self.vertex_texture = self.ctx.texture(size=(len(vertices),1), components=4, 
                                data=np.zeros((len(vertices)*4),dtype="f4", order='C'),
                                alignment=1,
                                dtype="f4")   
            '''                  
            print("cell texture size =",self.cell_texture.size,"components=",self.cell_texture.components)
            #exit()
            self.cell_framebuffer = self.ctx.framebuffer(color_attachments=self.cell_texture)
            #print("Framebuffer for cell: Size =",self.cell_framebuffer.size, "viewport =",self.cell_framebuffer.viewport)
            #self.cell_framebuffer.use()
            self.vbo = self.ctx.buffer(vertices)
            
            #print(self.vbo)
            self.fp_vao = self.ctx.vertex_array(
                self.cluster_quadric_map_generation_prog, 
                [
                    (self.vbo, '3f', 'inVert'),
                ],
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
            self.cell_framebuffer.use() 
            #print(self.ctx.fbo)
            self.ctx.clear(0., 0., 0., 0.)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = self.ctx.ADDITIVE_BLENDING # Required to add quadrics together
            self.fp_vao.render(mode=moderngl.POINTS)
            if not self.first_pass_output:
                self.first_pass_output = True

                num_components = 4
                '''
                print("Screen size:",self.ctx.screen.size, len(self.ctx.screen.read(components=num_components, dtype="f4")))
                #print(self.ctx.screen.read(components=num_components, dtype="f4"))
                raw_data = self.ctx.screen.read(components=num_components, dtype="f4")
                first_pass_data = np.frombuffer(raw_data, dtype="f4"),
                
                print("FP Data Shape:", first_pass_data[0].shape)
                new_shape = list(self.wnd.size) + [num_components]
                print("New shape:",new_shape )
                first_pass_data = np.reshape(first_pass_data, newshape=new_shape )     
                print("FP Data Shape:", first_pass_data.shape)
                '''
                                       
                #exit()
                print("Framebuffer size:",len(self.cell_framebuffer.read(components=4, dtype="f4")))
                print(self.cell_framebuffer.read(components=4, dtype="f4")[120:140])
                first_pass_data = np.reshape(
                    np.frombuffer(self.cell_framebuffer.read(components=4, dtype="f4"), dtype=np.float32),
                            newshape=(self.resolution**2, self.resolution * 4, 4)
                    )                
                print(first_pass_data.shape)
                #exit()
                np.save("first_pass_output_single_point", first_pass_data)
                #exit()
                print(os.path.getsize("./first_pass_output_single_point.npy"))
                self.first_pass_output = True
                
                self.first_pass = False
                #self.cell_framebuffer.release()
        elif not self.first_pass and self.mini_tris:
            #self.ctx.clear(self.back_color)
            self.ctx.clear(0.0, 0.0, 0.0)
            #self.vao.render(mode=moderngl.POINTS, vertices=100, instances=2)
            self.vao.render(mode=moderngl.POINTS)
            self.miniTrisProg['model'].value = tuple(transf.compose_matrix(angles=(0, np.pi/2 * run_time/8, 0)).ravel())

        else:
            self.close()
            exit()



        

if __name__ == '__main__':
    FirstPassWindow.run()
    #fp_window.run()
