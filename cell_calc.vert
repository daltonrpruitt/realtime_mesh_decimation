#version 430

uniform struct BoundingBox {
    vec3 min;
    vec3 max;
} bbox;

//out int inst;
in vec3 inVert;
out float cell_id;

uniform float cell_full_scale;
uniform float resolution;

uniform mat4 model; 
uniform mat4 view; 
uniform mat4 proj;


void main() {

    // Pass 1 - Step 1 : Determine Cell ID
    //      a. Shrink model to -1, 1
    vec3 avg = (bbox.min + bbox.max ) / 2.0;
    vec3 scale = (bbox.max - bbox.min) / 2.0;
    vec3 scaled_down_vert = (inVert - avg)/scale;
    //      b. Expand model to cell_full_scale
    vec3 scaled_up_vert = scaled_down_vert * cell_full_scale;
    //      c. Calc id based on scaled_up_vert / resolution 
    //                  x + y*resolution + z*resolution^2
    vec3 cell_indices = trunc(scaled_up_vert/resolution);
    cell_id = cell_indices.x + cell_indices.y * resolution + cell_indices.z * pow(resolution, 2.0);

    gl_Position = proj * view * model * vec4(scaled_down_vert, 0.0);
}