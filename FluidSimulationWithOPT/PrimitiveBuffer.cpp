#include "PrimitiveBuffer.h"

void Primitive::createSphereBuffers(float radius, int resolution)
{
	float PI = static_cast<float>(M_PI);
	// vectors to hold our data
	// vertice positions
	std::vector<Eigen::Vector3f> v;
	// normals
	std::vector<Eigen::Vector3f> n;
	std::vector<unsigned short> indices;

	// initiate the variable we are going to use
	float X1, Y1, X2, Y2, Z1, Z2;
	float inc1, inc2, inc3, inc4, radius1, radius2;

	for (int w = 0; w < resolution; w++)
	{
		for (int h = (-resolution / 2); h < (resolution / 2); h++)
		{
			inc1 = (w / (float)resolution) * 2 * PI;
			inc2 = ((w + 1) / (float)resolution) * 2 * PI;
			inc3 = (h / (float)resolution)*PI;
			inc4 = ((h + 1) / (float)resolution)*PI;

			X1 = sin(inc1);
			Y1 = cos(inc1);
			X2 = sin(inc2);
			Y2 = cos(inc2);

			// store the upper and lower radius, remember everything is going to be drawn as triangles
			radius1 = radius*cos(inc3);
			radius2 = radius*cos(inc4);

			Z1 = radius*sin(inc3);
			Z2 = radius*sin(inc4);

			// insert the triangle coordinates
			v.push_back(Eigen::Vector3f(radius1*X1, Z1, radius1*Y1));
			v.push_back(Eigen::Vector3f(radius1*X2, Z1, radius1*Y2));
			v.push_back(Eigen::Vector3f(radius2*X2, Z2, radius2*Y2));

			indices.push_back((unsigned short)v.size() - 3);
			indices.push_back((unsigned short)v.size() - 2);
			indices.push_back((unsigned short)v.size() - 1);

			v.push_back(Eigen::Vector3f(radius1*X1, Z1, radius1*Y1));
			v.push_back(Eigen::Vector3f(radius2*X2, Z2, radius2*Y2));
			v.push_back(Eigen::Vector3f(radius2*X1, Z2, radius2*Y1));

			indices.push_back((unsigned short)v.size() - 3);
			indices.push_back((unsigned short)v.size() - 2);
			indices.push_back((unsigned short)v.size() - 1);

			// insert the normal data
			n.push_back(Eigen::Vector3f(X1, Z1, Y1));
			n.push_back(Eigen::Vector3f(X2, Z1, Y2));
			n.push_back(Eigen::Vector3f(X2, Z2, Y2));
			n.push_back(Eigen::Vector3f(X1, Z1, Y1));
			n.push_back(Eigen::Vector3f(X2, Z2, Y2));
			n.push_back(Eigen::Vector3f(X1, Z2, Y1));
		}
	}

	for (unsigned int i = 0; i < n.size(); i++)
		n[i].normalize();


	glGenBuffersARB(1, &vertexbuffer);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexbuffer);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, v.size() * sizeof(Eigen::Vector3f), &v[0], GL_STATIC_DRAW);

	glGenBuffersARB(1, &normalbuffer);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalbuffer);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, n.size() * sizeof(Eigen::Vector3f), &n[0], GL_STATIC_DRAW);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	// Generate a buffer for the indices as well
	glGenBuffersARB(1, &elementbuffer);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, elementbuffer);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

	// store the number of indices for later use
	vertexBufferSize = (unsigned int)indices.size();

	// clean up after us
	indices.clear();
	n.clear();
	v.clear();
}
void Primitive::renderSphere(const Eigen::Vector3f &x, const float color[], const float transform[])
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexbuffer);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalbuffer);
	glNormalPointer(GL_FLOAT, 0, 0);

	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, elementbuffer);

	glPushMatrix();
	glTranslatef(x[0], x[1], x[2]);
	if (transform != NULL)
		glMultMatrixf(transform);
	glDrawElements(GL_TRIANGLES, (GLsizei)vertexBufferSize, GL_UNSIGNED_SHORT, 0);
	glPopMatrix();
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}
void Primitive::createBoxBuffers()
{
	// vertice positions
	std::vector<Vector3f> v;
	// normals
	std::vector<Vector3f> n;
	std::vector<unsigned short> indices;

	// initiate the variable we are going to use
	float X1, Y1, X2, Y2, Z1, Z2;
	float inc1, inc2, inc3, inc4, radius1, radius2;

	// 0, 1, 2, 3
	v.push_back(Vector3f(-1.0, 1.0, -1.0));
	v.push_back(Vector3f(1.0, 1.0, -1.0));
	v.push_back(Vector3f(-1.0, 1.0, 1.0));
	v.push_back(Vector3f(1.0, 1.0, 1.0));
	// 4, 5, 6, 7
	v.push_back(Vector3f(-1.0, -1.0, -1.0));
	v.push_back(Vector3f(1.0, -1.0, -1.0));
	v.push_back(Vector3f(-1.0, -1.0, 1.0));
	v.push_back(Vector3f(1.0, -1.0, 1.0));

	n.push_back(Vector3f(X2, Z1, Y2));
	n.push_back(Vector3f(X2, Z2, Y2));

	indices.push_back(0);
	indices.push_back(1);
	indices.push_back(2);
	n.push_back(Vector3f(0, 1, 0));
	indices.push_back(1);
	indices.push_back(3);
	indices.push_back(2);
	n.push_back(Vector3f(0, 1, 0));

	indices.push_back(1);
	indices.push_back(7);
	indices.push_back(3);
	n.push_back(Vector3f(1, 0, 0));
	indices.push_back(1);
	indices.push_back(5);
	indices.push_back(7);
	n.push_back(Vector3f(1, 0, 0));

	indices.push_back(0);
	indices.push_back(2);
	indices.push_back(6);
	n.push_back(Vector3f(-1, 0, 0));
	indices.push_back(0);
	indices.push_back(6);
	indices.push_back(4);
	n.push_back(Vector3f(-1, 0, 0));

	indices.push_back(0);
	indices.push_back(4);
	indices.push_back(1);
	n.push_back(Vector3f(0, 0, -1));
	indices.push_back(1);
	indices.push_back(4);
	indices.push_back(5);
	n.push_back(Vector3f(0, 0, -1));

	indices.push_back(2);
	indices.push_back(3);
	indices.push_back(6);
	n.push_back(Vector3f(0, 0, 1));
	indices.push_back(3);
	indices.push_back(7);
	indices.push_back(6);
	n.push_back(Vector3f(0, 0, 1));

	indices.push_back(4);
	indices.push_back(6);
	indices.push_back(7);
	n.push_back(Vector3f(0, -1, 0));
	indices.push_back(4);
	indices.push_back(7);
	indices.push_back(5);
	n.push_back(Vector3f(0, -1, 0));

	glGenBuffersARB(1, &vertexbuffer);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexbuffer);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, v.size() * sizeof(Vector3f), &v[0], GL_STATIC_DRAW);

	glGenBuffersARB(1, &normalbuffer);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalbuffer);
	glBufferDataARB(GL_ARRAY_BUFFER_ARB, n.size() * sizeof(Vector3f), &n[0], GL_STATIC_DRAW);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	// Generate a buffer for the indices as well
	glGenBuffersARB(1, &elementbuffer);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, elementbuffer);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER_ARB, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

	// store the number of indices for later use
	vertexBufferSize = (unsigned int)indices.size();

	// clean up after us
	indices.clear();
	n.clear();
	v.clear();
}
void Primitive::renderBox(Vector3f position, Vector3f scale, float color[])
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, vertexbuffer);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalbuffer);
	glNormalPointer(GL_FLOAT, 0, 0);

	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, elementbuffer);

	glPushMatrix();
	glTranslatef(position[0], position[1], position[2]);
	glScalef(scale[0], scale[1], scale[2]);
	glDrawElements(GL_TRIANGLES, (GLsizei)vertexBufferSize, GL_UNSIGNED_SHORT, 0);
	glPopMatrix();
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}
void Primitive::releaseBuffers()
{
	if (elementbuffer != 0)
	{
		glDeleteBuffersARB(1, &elementbuffer);
		elementbuffer = 0;
	}
	if (normalbuffer != 0)
	{
		glDeleteBuffersARB(1, &normalbuffer);
		normalbuffer = 0;
	}
	if (vertexbuffer != 0)
	{
		glDeleteBuffersARB(1, &vertexbuffer);
		vertexbuffer = 0;
	}
}