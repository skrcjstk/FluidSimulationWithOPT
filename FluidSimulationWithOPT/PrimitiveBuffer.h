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
	void createSphereBuffers(float& radius, int resolution);
	void renderSphere(Vector3f &x, float color[], float transform[] = NULL);
	void createBoxBuffers();
	void renderBox(Vector3f& position, Vector3f& scale, float color[]);
	void createWireFrameBoxBuffers();
	void renderWireFrameBox(Vector3f& position, Vector3f& scale, float color[]);
	void releaseBuffers();
	void renderArrow3D(Vector3f& start, Vector3f& end, float& arrow_head_len);

private:
	GLuint elementbuffer;
	GLuint normalbuffer;
	GLuint vertexbuffer;
	int vertexBufferSize = 0;

	// vertice positions
	std::vector<Vector3f> v;
	// normals
	std::vector<Vector3f> n;
	std::vector<unsigned short> indices;
};

#endif 
