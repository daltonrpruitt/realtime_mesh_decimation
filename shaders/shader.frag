#version 430

uniform vec4 in_color;

in vec3 normal;
out vec4 f_color;

vec3 light_pos = vec3(0.5, 1.0, 0.0);

void main() {

    float lambertian = dot(-normal, normalize(light_pos-gl_FragCoord.xyz));
    f_color = (0.5 + 0.5* lambertian) * in_color;
    
}

