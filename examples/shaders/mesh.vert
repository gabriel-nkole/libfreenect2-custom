#version 330 core

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec4 inColor;

uniform mat4 WVP;

out vec4 color;

void main() {
	gl_Position = WVP * vec4(inPos, 1.0f);
	color = inColor;
}