#version 430

//in vec3 fragColor;
in float cell_id_geom;

out vec4 f_color;

void main() {

    f_color = vec4(cell_id_geom);
    
}

