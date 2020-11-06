#version 430

uniform struct BoundingBox {
    vec3 min;
    vec3 max;
} bbox;

//out int inst;
in vec3 inVert;

uniform mat4 model; 
uniform mat4 view; 
uniform mat4 proj;


void main() {
    vec3 avg = (bbox.min + bbox.max )/ 2.0;
    vec3 scale = (bbox.max - bbox.min)/1.5;
    gl_Position = proj * view * model * vec4((inVert - avg)/max(scale.x, max(scale.y, scale.z)), 1.0);
}