#version 430

layout(triangles) in;
layout (triangle_strip, max_vertices = 3) out;

//in vec3 inColor[];

out vec3 fragColor;

uniform mat4 model; 
uniform mat4 view; 
uniform mat4 proj;

void main(void) {
    vec4 offset = vec4(-1.0, 1.0, 0.0, 0.0);
    vec4 vertexPos = offset + gl_in[0].gl_Position;
    gl_Position = proj * view * model * vertexPos;
    fragColor = vec3(1., 1., 1.) * vec3(1.0, 0.0, 0.0);

}