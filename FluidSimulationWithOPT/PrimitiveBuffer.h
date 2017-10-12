#pragma once
#ifndef __PRIMITIVEBUFFER_H__
#define __PRIMITIVEBUFFER_H__

#include "GL/glew.h"
#include "Visualization\MiniGL.h"
#include "Visualization\Selection.h"
#include "GL/glut.h"
#include <Eigen/Dense>

using namespace Eigen;

class Primitive
{
public:
	void createSphereBuffers(float radius, int resolution);
	void renderSphere(const Eigen::Vector3f &x, const float color[], const float transform[] = NULL);
	void createBoxBuffers();
	void renderBox(Vector3f position, Vector3f scale, float color[]);
	void releaseBuffers();

private:
	GLuint elementbuffer;
	GLuint normalbuffer;
	GLuint vertexbuffer;
	int vertexBufferSize = 0;
};

#endif 
