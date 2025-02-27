#version 330 core
precision highp float;

in vec4 color;

out vec4 FragColor;

void main() {
	//if (color.x < 0.1f)
	//	discard;
	FragColor = color;
}