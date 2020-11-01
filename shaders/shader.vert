#version 330
uniform struct BoundingBox {
    vec3 min;
    vec3 max;
} bbox;

//out int inst;
in vec3 inVert;


void main() {
    vec3 avg = (bbox.min + bbox.max )/ 2.0;
    vec3 scale = (bbox.max - bbox.min)/1.5;
    gl_Position = vec4((inVert - avg)/scale, 0.0);
}