#version 430
layout (triangles) in;
layout (points, max_vertices = 3) out;

in float cell_id[3];
out float cell_id_geom;

in int inst[3];

void main() {

    //float vert1 = gl_in[0].gl_Position.xyz;
    gl_Position = vec4(gl_in[0].gl_Position.x, gl_in[0].gl_Position.y * gl_in[0].gl_Position.z, 0., 0.) ; //vec4(x - 0.007, y - 0.007, 0.0, 1.0);
    cell_id_geom = cell_id[0];
    EmitVertex();
    EndPrimitive();

    gl_Position = vec4(gl_in[1].gl_Position.x, gl_in[1].gl_Position.y * gl_in[1].gl_Position.z, 0., 0.) ; //vec4(x - 0.007, y - 0.007, 0.0, 1.0);
    cell_id_geom = cell_id[1];
    EmitVertex();
    EndPrimitive();

    gl_Position = vec4(gl_in[2].gl_Position.x, gl_in[2].gl_Position.y * gl_in[2].gl_Position.z, 0., 0.) ; //vec4(x - 0.007, y - 0.007, 0.0, 1.0);
    cell_id_geom = cell_id[2];
    EmitVertex();
    EndPrimitive();
}
