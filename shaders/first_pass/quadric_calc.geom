#version 430
layout (points) in;
layout (points, max_vertices = 1) out;

in float cell_id[1];
out float cell_id_geom;

uniform float resolution;

in int inst[1];

void main() {

    /*
    float x = gl_in[0].gl_Position.x;
    float y = gl_in[0].gl_Position.y;
    //float x = float(gl_PrimitiveIDIn / 10) / 9 - 0.5 + inst[0] / 20.0;
    //float y = float(gl_PrimitiveIDIn % 10) / 9 - 0.5 + inst[0] / 20.0;
    gl_Position = vec4(x - 0.007, y - 0.007, 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(x + 0.007, y - 0.007, 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(x, y + 0.007, 0.0, 1.0);
    EmitVertex();
    */
    float x_range = resolution * resolution;
    float y_range = resolution * 4;
    //float vert1 = gl_in[0].gl_Position.xyz;
    //gl_Position = vec4(gl_in[0].gl_Position.x * gl_in[0].gl_Position.z, gl_in[0].gl_Position.y, 0., 0.) ; //vec4(x - 0.007, y - 0.007, 0.0, 1.0);
    cell_id_geom = cell_id[0];
    gl_Position = vec4( (2.0 * mod(cell_id[0], x_range) - x_range)/x_range,
                         -1.0 * (2.0 * 4.0 * trunc(cell_id[0] / (resolution * resolution)) - y_range) / y_range , 
                         0.0, 1.0);
    EndPrimitive();
    /*
    gl_Position = vec4(x-0.005, y-0.005, 0.0, 1.0);
    EmitVertex();

    gl_Position =  vec4(x+0.005, y-0.005, 0.0, 1.0);
    EmitVertex();

    gl_Position = vec4(x, y+0.005, 0.0, 1.0);
    EmitVertex();

    EndPrimitive();
    */

    /*
    //gl_Position = vec4(gl_in[1].gl_Position.x * gl_in[1].gl_Position.z, gl_in[1].gl_Position.y, 0., 0.) ; //vec4(x - 0.007, y - 0.007, 0.0, 1.0);
    gl_Position = vec4( mod(cell_id[1], resolution * resolution), 4.0 * trunc(cell_id[1] / (resolution * resolution)), 0.0, 0.0);
    cell_id_geom = cell_id[1];
    EmitVertex();
    EndPrimitive();

    //gl_Position = vec4(gl_in[2].gl_Position.x * gl_in[2].gl_Position.z, gl_in[2].gl_Position.y, 0., 0.) ; //vec4(x - 0.007, y - 0.007, 0.0, 1.0);
    gl_Position = vec4( mod(cell_id[2], resolution * resolution), 4.0 * trunc(cell_id[2] / (resolution * resolution)), 0.0, 0.0);
    cell_id_geom = cell_id[2];
    EmitVertex();
    EndPrimitive();
    */
}
