#version 330 core

// vertex
layout (location = 0) in vec3 inPos;

// instance
layout (location = 1) in vec4 inColor;
layout (location = 2) in mat4 inWVP;

out vec4 color;

void main() {
	gl_Position = inWVP * vec4(inPos, 1.0f);
	color = inColor;
}