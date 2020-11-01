#version 430
layout (points) in;
layout (triangle_strip, max_vertices = 3) out;
in int inst[1];
void main() {
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
    EndPrimitive();
}
