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

# Directly from moderngl compute_shader.py example : https://github.com/moderngl/moderngl/blob/master/examples/compute_shader.py
def source(uri, consts):
    ''' read gl code '''
    with open(uri, 'r') as fp:
        content = fp.read()

    # feed constant values
    for key, value in consts.items():
        content = content.replace(f"%%{key}%%", str(value))
    return content

debug = False
resolution = 2
float_to_int_scaling_factor = 2**13


st = time.time()
#self.obj_mesh = trimesh.exchange.load.load("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj")
obj_mesh = pywavefront.Wavefront("meshes/ssbb-toon-link-obj/DolToonlinkR1_fixed.obj", collect_faces=True)
print("Loading mesh took {:.2f} s".format(time.time()-st))
#vertices = np.array(obj_mesh.vertices, dtype='f4')
#bbox = utility.bounding_box(vertices) # Makes bounding box
indices = np.array(obj_mesh.mesh_list[0].faces)
vertices = np.array(utility.positive_vertices(obj_mesh.vertices),dtype="f4")
if debug:
    print("Input Vertices:")
    print("\tMin,max of x:",min(vertices[:,0]),",",max(vertices[:,0]))
    print("\tMin,max of y:",min(vertices[:,1]),",",max(vertices[:,1]))
    print("\tMin,max of z:",min(vertices[:,2]),",",max(vertices[:,2]))

st = time.time()
vertex_cluster_indices, vertex_cluster_ids = get_vertex_cell_indices_ids(vertices=vertices, resolution=resolution)
print("Generating vertex cluster indicies and IDs took {:.2f} s".format(time.time()-st))
vertex_cluster_ids = np.array(vertex_cluster_ids, dtype=np.int32)

print(vertex_cluster_ids[:30])
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
print("Successfully compiled compute shader!")


# Create/bind vertex/index data
vertex_buffer = fp_context.buffer(vertices)
vertex_buffer.bind_to_storage_buffer(binding=0)
index_buffer = fp_context.buffer(indices)
index_buffer.bind_to_storage_buffer(binding=1)

cluster_id_buffer = fp_context.buffer(vertex_cluster_ids)
cluster_id_buffer.bind_to_storage_buffer(binding=2)

#print(np.reshape(np.frombuffer(vertex_buffer.read(),dtype="f4"),newshape=(len(vertices),3))[:5])

image_shape = (size, 14)
# Output "image" creation
cluster_quadric_map_int = fp_context.texture(size=image_shape, components=1, dtype="i4")
cluster_quadric_map_int.bind_to_image(4, read=True, write=True)

#quadric_map = fp_context.texture(size=(size,4), components=4,dtype="f4") 

first_pass_comp_shader['resolution'] = resolution
first_pass_comp_shader['float_to_int_scaling_factor'] = float_to_int_scaling_factor


st = time.time()
first_pass_comp_shader.run(size, 1, 1)
print("Running FP Compute Shader Took {:.5f} s".format(time.time()-st))

# Output "image" reading
output_data_original = np.frombuffer(cluster_quadric_map_int.read(),dtype=np.int32)
#print(output_data[:28])
output_data = np.reshape(output_data_original, newshape=image_shape, order="F") / float_to_int_scaling_factor
if debug:
    print(output_data[55:57,:4])
    print(sum(output_data[:,3]))
    print("Actual num of tris:", len(indices))


output_array = np.array(output_data, dtype=np.float32)
'''
iter, span = 0, 20
while True:
    print(iter,"to",iter+span, ":", output_array[iter:iter+span])
    iter+=span
    if input("Enter nothing to continue:") != "":
        break'''
if debug:
    print(output_array[:2][:])
    output_sum_vertices = output_array[:,:3]
    print(output_sum_vertices.shape)
    output_count_vertices = output_array[:,3]
    
    avg_vertices = np.empty(shape=(1,3),dtype=np.float32)
    for i in range(len(output_sum_vertices)):
        if output_count_vertices[i] > 0.1:
            avg_vertices = np.concatenate((avg_vertices, np.ndarray(shape=(1,3),
                                buffer=np.array([output_sum_vertices[i][j]/output_count_vertices[i] for j in range(3)]),
                                dtype=np.float32)))
    avg_vertices = avg_vertices[1:]
    print("Avg. Vertices:", avg_vertices.shape)
    print("min,max x:",min(avg_vertices[:,0]), max(avg_vertices[:,0]))
    print("min,max y:",min(avg_vertices[:,1]), max(avg_vertices[:,1]))
    print("min,max z:",min(avg_vertices[:,2]), max(avg_vertices[:,2]))


# End of First Pass

##########################################

sp_context = moderngl.create_standalone_context(require=430)
second_pass_comp_shader = sp_context.compute_shader(source("shaders/secondpass.comp", shader_constants))
print("Successfully compiled second pass compute shader!")


sp_cluster_quadric_map = sp_context.texture(size=image_shape,components=1, 
                                            data=output_data_original, dtype="i4")
if debug:
    print(np.frombuffer(sp_cluster_quadric_map.read(),dtype=np.int32)[55*14:57*14])
    print(output_data_original[55*14:57*14])

sp_cluster_quadric_map.bind_to_image(0, read=True, write=False)

sp_cluster_vertex_positions = sp_context.texture(size=(image_shape[0],1),components=4, 
                                            data=None, dtype="f4")
sp_cluster_vertex_positions.bind_to_image(1, read=False, write=True)


second_pass_comp_shader['float_to_int_scaling_factor'] = float_to_int_scaling_factor
#second_pass_comp_shader['resolution'] = resolution

st = time.time()
second_pass_comp_shader.run(size, 1, 1)
print("Running SP Compute Shader Took {:.5f} s".format(time.time()-st))

# Second Pass Output "image" reading
sp_output_vertex_positions = np.frombuffer(sp_cluster_vertex_positions.read(),dtype=np.float32)
sp_output_vertex_positions = np.reshape(sp_output_vertex_positions, newshape=(image_shape[0],4), order="C")

if True:
    debug_vertex_positions = sp_output_vertex_positions[sp_output_vertex_positions[:,3] > -1.0]
    print("Without empty cells:", sp_output_vertex_positions.shape,"-->",debug_vertex_positions.shape)
    print("\tNon-empty cells:", str(round(debug_vertex_positions.shape[0]/sp_output_vertex_positions.shape[0]*100,2))+"%")
    print("Min,max of x:",min(debug_vertex_positions[:,0]),",",max(debug_vertex_positions[:,0]))
    print("Min,max of y:",min(debug_vertex_positions[:,1]),",",max(debug_vertex_positions[:,1]))
    print("Min,max of z:",min(debug_vertex_positions[:,2]),",",max(debug_vertex_positions[:,2]))
    print(debug_vertex_positions[:resolution*4,:3])

exit()

# End of Second Pass

##########################################
# Third Pass
tp_context = moderngl.create_standalone_context(require=430)
tp_compute_shader = tp_context.compute_shader(source("shaders/thirdpass.comp", shader_constants))
print("Successfully compiled third pass compute shader!")


sp_cluster_quadric_map = sp_context.texture(size=image_shape,components=1, 
                                            data=output_data_original, dtype="i4")
if debug:
    print(np.frombuffer(sp_cluster_quadric_map.read(),dtype=np.int32)[55*14:57*14])
    print(output_data_original[55*14:57*14])



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


class RenderWindow(BasicWindow):
    gl_version = (4, 3)
    title = "Geometry Shader Testing"

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.back_color = (0, 0.3, 0.9, 1.0) #(1,1,1, 1)


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
    RenderWindow.run()
    #fp_window.run()
