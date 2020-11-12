#version 430

layout(triangles) in;
layout (triangle_strip, max_vertices = 3) out;

//in vec3 inColor[];

//out vec3 fragColor;
out vec3 normal; 

uniform mat4 model; 
uniform mat4 view; 
uniform mat4 proj;


void main(void) {

    vec4 verts[3] = {gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position};

    // Assumed CCW front face
    vec3 norm = normalize(cross(verts[1].xyz-verts[0].xyz, verts[2].xyz-verts[0].xyz));

    gl_Position = verts[0];
    normal = norm;
    EmitVertex();

    gl_Position = verts[1];
    normal = norm;
    EmitVertex();

    gl_Position = verts[2];
    normal = norm;
    EmitVertex();

    EndPrimitive();
}