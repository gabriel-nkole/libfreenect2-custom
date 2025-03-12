#version 330 core
precision highp float;

in vec4 color;
in vec3 normal;

out vec4 FragColor;

void main() {
	if (color.w < 0.1f)
		discard;
	FragColor = color;
	//FragColor = vec4(normal, 1.0f);
	//FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}