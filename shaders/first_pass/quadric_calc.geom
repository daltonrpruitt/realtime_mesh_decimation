#version 430
layout (points) in;
layout (points, max_vertices = 1) out;

in float cell_id[1];
out float cell_id_geom;
out vec4 g_color;

out vec3 positive_vertex_position;

uniform float resolution;

in int inst[1];

void main() {

    /* Layout of the data for each cell
    *  Row 1: sum_vertex_position [x, y, z], vertex_count (together -> average vertex position)
    *  Row 2: a^2, ab, ac, ad
    *  Row 3: b^2, bc, bd, c^2
    *  Row 4: cd, d^2, cell_id, -1.0
    * 
    *  Columns are just iterating up each cell_ID
    *
    *  Cell_0[0]                    Cell_1[0]                   Cell_2[0]                   Cell_3[0]                   ...      Cell_res^2-1[0]
    *  Cell_0[1]                    Cell_1[1]                   Cell_2[1]                   Cell_3[1]                   ...      Cell_res^2-1[1]
    *  ...                          ...                         ...                         ...                         ...      ...
    *  Cell_0[4]                    Cell_1[4]                   Cell_2[4]                   Cell_3[4]                   ...      Cell_res^2-1[4]
    *  Cell_res^2[0]                Cell_res^2+1[0]             Cell_res^2+1[0]             Cell_res^2+3[0]             ...      Cell_res^2-1[0]
    *  ...                          ...                         ...                         ...                         ...      ...
    *  Cell_res^2(res-1)[4]         Cell_res^2(res-1)+1[4]      Cell_res^2(res-1)+2[4]      Cell_res^2(res-1)+2[4]       ...     Cell_res^3-1[4]
    *
    *  For a total of (res^2) * (res * 4) = 4res^3 "pixel" location in the array
    *
    */
   
    //float x = gl_in[0].gl_Position.x;
    //float y = gl_in[0].gl_Position.y;
    float x_range = resolution * resolution;
    float y_range = resolution * 4.0;

    //float x = float(gl_PrimitiveIDIn / 10) / 9 - 0.5 + inst[0] / 20.0;
    //float y = float(gl_PrimitiveIDIn % 10) / 9 - 0.5 + inst[0] / 20.0;
    //gl_Position = vec4(x - 0.05, y - 0.05, 0.0, 1.0);

    gl_Position = vec4(gl_in[0].gl_Position.xy * 0.0001, 1.0, 1.0);/*vec4( (2.0 * mod(cell_id[0], x_range) - x_range)/x_range,
                        -1.0 * (2.0 * 4.0 * trunc(cell_id[0] / x_range) - y_range + 0.0 * 2.0 / y_range) / y_range , 
                        0.0, 1.0);*/
    g_color = vec4(gl_in[0].gl_Position.xyz, 1.0);

    EmitVertex();
     /*
     gl_Position = vec4(x + 0.05, y - 0.05, 0.0, 1.0);
    EmitVertex();
    gl_Position = vec4(x, y + 0.05, 0.0, 1.0);
    EmitVertex();
    
    float x_range = resolution * resolution;
    float y_range = resolution * 4;
    //float vert1 = gl_in[0].gl_Position.xyz;
    //gl_Position = vec4(gl_in[0].gl_Position.x * gl_in[0].gl_Position.z, gl_in[0].gl_Position.y, 0., 0.) ; //vec4(x - 0.007, y - 0.007, 0.0, 1.0);
    cell_id_geom = cell_id[0];
    gl_Position = vec4( (2.0 * mod(cell_id[0], x_range) - x_range)/x_range,
                         -1.0 * (2.0 * 4.0 * trunc(cell_id[0] / (resolution * resolution)) - y_range) / y_range , 
                         0.0, 1.0);
    */
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
