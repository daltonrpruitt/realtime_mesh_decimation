#version 430

uniform struct BoundingBox {
    vec3 min;
    vec3 max;
} bbox;

//out int inst;
in vec3 inVert;
out float cell_id;

//uniform float cell_full_scale;
uniform float resolution;

uniform mat4 model; 
uniform mat4 view; 
uniform mat4 proj;


void main() {

    // Pass 1 - Step 1 : Determine Cell ID
    //      a. Put model in only positive space
    vec3 avg = (bbox.min + bbox.max ) / 2.0;
    vec3 scale = (bbox.max - bbox.min) ;
    //vec3 scaled_down_vert = (inVert - avg)/scale;
    vec3 pos_only_vert = inVert + -bbox.min;
    //      b. Expand model to resolution^2 size
    vec3 scaled_up_vert = pos_only_vert * pow(resolution, 2.0) / (scale+0.00001);
    //      c. Calc id based on scaled_up_vert / resolution 
    //                  x + y*resolution + z*resolution^2
    vec3 cell_indices = trunc(scaled_up_vert/resolution);
    
    cell_id = cell_indices.x + cell_indices.y * resolution + cell_indices.z * pow(resolution, 2.0);

    //gl_Position = vec4(cell_indices/(resolution/2.0) - 1, 0.0);// proj * view * model * vec4(scaled_down_vert, 0.0);
    
    // The positive version of the vertices is proabably the most convenient to send onwards, as its
    //  scale is the same as the input data, and I don't like negatives
    gl_Position = vec4(pos_only_vert, 1.); 
    
    
    //gl_Position = vec4((inVert - avg)/max(scale.x, max(scale.y, scale.z)), 0.0);

    //float x_range = resolution * resolution;
    //float y_range = resolution * 4;
    //float vert1 = gl_in[0].gl_Position.xyz;
    //gl_Position = vec4(gl_in[0].gl_Position.x * gl_in[0].gl_Position.z, gl_in[0].gl_Position.y, 0., 0.) ; //vec4(x - 0.007, y - 0.007, 0.0, 1.0);
    //cell_id_geom = cell_id[0];

    //gl_Position = vec4( (2.0 * mod(cell_id, x_range) - x_range)/x_range,
      //                   -1.0 * (2.0 * 4.0 * trunc(cell_id / (resolution * resolution)) - y_range) / y_range , 
        //                 0.0, 1.0);

}