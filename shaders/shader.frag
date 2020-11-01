#version 430

in vec3 fragColor;

out vec4 f_color;

void main() {

    f_color = vec4(fragColor, 1.0);
    
}

