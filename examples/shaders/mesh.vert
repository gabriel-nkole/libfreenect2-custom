#version 330 core

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec4 inColor;
layout (location = 2) in vec3 inNormal;

uniform mat4 WVP;

out vec4 color;
out vec3 normal;

void main() {
	gl_Position = WVP * vec4(inPos, 1.0f);
	color = inColor;
	normal = inNormal;
}