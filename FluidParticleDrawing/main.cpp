#include "GL/glew.h"
#include "..\FluidSimulationWithOPT\Visualization\MiniGL.h"
#include "..\FluidSimulationWithOPT\Visualization\Selection.h"
#include "GL/glut.h"
#include "..\FluidSimulationWithOPT\Visualization\PrimitiveBuffer.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace PBD;
using namespace std;
using namespace Eigen;

string positionDataPath = "E:/Tensorflow/tensorflow/tensorflow/contrib/cmake/build/modelLoader/PBFCTrainSD11_180102/";
//string positionDataPath = "../FluidSimulationWithOPT/PBFCDataScene/v0.01/";
int frameLimit = 700;

GLint context_major_version, context_minor_version;
Primitive spherePrimiCoarse, spherePrimiFine, boxPrimi;

void timeStep();
void reset();
void render();
void cleanup();
void DataLoad();

float fineR = 0.025f;
float coarseR = 2 * fineR;
bool doPause = true;
int accFrameCount = 0;

std::vector<Vector3f> subWorld;
std::vector<Vector3f> mainWorld;

int main(int argc, char** argv)
{
	// OpenGL	
	MiniGL::init(argc, argv, 1024, 768, 0, 0, "Fluid demo");
	MiniGL::initLights();
	MiniGL::setClientIdleFunc(50, timeStep);
	MiniGL::setKeyFunc(0, 'r', reset);
	//MiniGL::setSelectionFunc(selection);

	MiniGL::getOpenGLVersion(context_major_version, context_minor_version);

	MiniGL::setClientSceneFunc(render);
	MiniGL::setViewport(40.0f, 0.1f, 500.0f, Vector3f(0.0, 2.0, 8.0), Vector3f(0.0, 2.0, 0.0));

	TwAddVarRW(MiniGL::getTweakBar(), "Pause", TW_TYPE_BOOLCPP, &doPause, " label='Pause' group=Simulation key=SPACE ");

	if (context_major_version >= 3)
	{
		spherePrimiCoarse.createSphereBuffers((float)coarseR, 8);
		spherePrimiFine.createSphereBuffers((float)fineR, 8);
		boxPrimi.createBoxBuffers();
	}

	glutMainLoop();

	cleanup();

	return 0;
}

void timeStep()
{
	if (doPause)
		return;

	DataLoad();
	accFrameCount++;
	if (accFrameCount > frameLimit)
	{
		doPause = !doPause;
	}
}

void reset() { accFrameCount = 0; }

void selection(const Eigen::Vector2i &start, const Eigen::Vector2i &end) {}
void render()
{
	MiniGL::coordinateSystem();
	MiniGL::drawTime(0);

	float surfaceColor[4] = { 0.2f, 0.2f, 0.2f, 0.1f };
	float kernelColor[4] = { 1.0f, 0.2f, 0.2f, 1.0f };
	float speccolor[4] = { 1.0, 1.0, 1.0, 1.0 };
	float anisotropyColor[4] = { 0.2f, 0.2f, 0.2f, 1.0f };

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, surfaceColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, surfaceColor);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, speccolor);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 100.0);
	glColor3fv(surfaceColor);

	glPointSize(4.0f);

	// drawing main particles
	float fluidColor[4] = { 0.0f, 0.7f, 0.7f, 0.2f };
	for (int i = 0; i <mainWorld.size(); i++)
	{
		spherePrimiCoarse.renderSphere(mainWorld[i], fluidColor);
	}

	Vector3f translation(0.0f, 1.5f, 0.0f);
	// drawing sub particles
	float subFluidColor[4] = { 0.2f, 0.3f, 0.9f, 0.8f };

	for (int i = 0; i < subWorld.size(); i++)
	{
		Vector3f pos = subWorld[i] + translation;
		spherePrimiFine.renderSphere(pos, subFluidColor);
	}
}

void cleanup()
{
	if (context_major_version >= 3)
	{
		spherePrimiCoarse.releaseBuffers();
		spherePrimiFine.releaseBuffers();
		boxPrimi.releaseBuffers();
	}
}

void DataLoad()
{
	int ibuf[1];
	float fbuf[3];

	string filename = positionDataPath + "MainWorld" + std::to_string(accFrameCount) + ".dat";
	FILE* fp = fopen(filename.c_str(), "rb");

	fread(ibuf, sizeof(int), 1, fp);
	if (mainWorld.size() == 0)
	{
		mainWorld.resize(ibuf[0]);
		for (int i = 0; i < ibuf[0]; i++)
			mainWorld[i] = Vector3f(0, 0, 0);
	}
	for (int i = 0; i < ibuf[0]; i++)
	{
		fread(fbuf, sizeof(float), 3, fp);
		mainWorld[i][0] = fbuf[0];	mainWorld[i][1] = fbuf[1];	mainWorld[i][2] = fbuf[2];
	}
	fclose(fp);

	filename = positionDataPath + "SubWorld" + std::to_string(accFrameCount) + ".dat";
	fp = fopen(filename.c_str(), "rb");

	fread(ibuf, sizeof(int), 1, fp);
	if (subWorld.size() == 0)
	{
		subWorld.resize(ibuf[0]);
		for (int i = 0; i < ibuf[0]; i++)
			subWorld[i] = Vector3f(0, 0, 0);
	}
	for (int i = 0; i < ibuf[0]; i++)
	{
		fread(fbuf, sizeof(float), 3, fp);
		subWorld[i][0] = fbuf[0];	subWorld[i][1] = fbuf[1];	subWorld[i][2] = fbuf[2];
	}
	fclose(fp);
	printf("%d frame's positionData were loaded.\n", accFrameCount);
}