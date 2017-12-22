#include "GL/glew.h"
#include "..\FluidSimulationWithOPT\Visualization\MiniGL.h"
#include "..\FluidSimulationWithOPT\Visualization\Selection.h"
#include "GL/glut.h"
#include "..\FluidSimulationWithOPT\PrimitiveBuffer.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "APICSim.h"
#include "..\PositionBasedFluid\FluidWorld.h"
#include "..\PositionBasedFluid\TimerChrono.h"

using namespace PBD;
using namespace std;
using namespace Eigen;


void timeStep();
void reset();
void render();
void cleanup();
void buildModel();

void CreateCoarseBreakingDam(std::vector<Vector3f>& p_damParticles);
void CreateCoarseContainer(std::vector<Vector3f>& p_boundaryParticles);
void AddWall(Vector3f p_min, Vector3f p_max, std::vector<Vector3f>& p_boundaryParticle, float p_particleRadius);

TimerChrono timer1;

float fineR = 0.025f;
float coarseR = 2 * fineR;
int fineDamWidth = 20;
int fineDamHeight = 20;
int fineDamDepth = 20;
int coarseDamWidth = fineDamWidth / 2;
int coarseDamHeight = fineDamHeight / 2;
int coarseDamDepth = fineDamDepth / 2;
float containerWidth = (coarseDamWidth + 1) * coarseR * 2.0f * 5.0f;
float containerHeight = 3.0f;
float containerDepth = (coarseDamDepth + 1) * coarseR * 2.0f;
Vector3f containerStart;
Vector3f containerEnd;

bool doPause = true;
int accFrameCount = 0;

APICSim sim;
FluidWorld* world;

int frameLimit = 1600;
GLint context_major_version, context_minor_version;
Primitive spherePrimiCoarse, spherePrimiFine, boxPrimi;

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
		boxPrimi.createWireFrameBoxBuffers();
	}

	buildModel();

	glutMainLoop();

	cleanup();

	return 0;
}

void timeStep()
{
	if (doPause)
		return;

	// particle simulation
	timer1.start();
	world->StepPBF();
	timer1.end("Simulation");

	timer1.start();
	sim.AssignCells(world);
	sim.UpdateAffineMatrix(world);
	sim.Map_P2G(world);
	timer1.end("Assign & Mapping");
	
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
	for (int i = 0; i <world->GetNumOfParticles(); i++)
	{
		spherePrimiCoarse.renderSphere(world->GetParticle(i)->m_curPosition, fluidColor);
	}

	float boxColor[4] = { 0.1f, 0.1f, 0.1f, 1.0f };
	
	// draw grid & arrow
	Vector3f nijk = sim.GetNiNjNk();
	Vector3f dxyz = sim.GetDxDyDz();
	float head_len = 0.5f*dxyz[0];
	for (int k=0; k<nijk[2]; k++)
		for (int j = 0; j<nijk[1]; j++)
			for (int i = 0; i<nijk[0]; i++)
			{
				Vector3f pos = sim.GetGridPos(i, j, k);
				//boxPrimi.renderWireFrameBox(pos, dxyz, boxColor);
				Vector3f end = (pos + 0.1f * sim.GetVelocity(pos));
				boxPrimi.renderArrow3D(pos, end, head_len);
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

void buildModel()
{
	std::vector<Vector3f> boundaryParticles;
	std::vector<Vector3f> damParticles;
	CreateCoarseBreakingDam(damParticles);
	CreateCoarseContainer(boundaryParticles);

	world = new FluidWorld();
	world->SetTimeStep(0.005f);
	world->CreateParticles(damParticles, boundaryParticles, coarseR);

	// origin=(0,0), gridWidth=100, grid_res=(100,100), rho=1.0
	Vector3f bSize = containerEnd - containerStart;
	sim.Initialize(containerStart, containerEnd - containerStart, Vector3i((int)(bSize[0] * 10), (int)(bSize[1] * 10), (int)(bSize[2] * 10)), 1.0);


}

void CreateCoarseBreakingDam(std::vector<Vector3f>& p_damParticles)
{
	std::cout << "Initialize coarse fluid particles\n";
	p_damParticles.resize(coarseDamWidth*coarseDamHeight*coarseDamDepth);

	float diam = 2.0f * coarseR;
	float startX = -0.5f * containerWidth + diam + diam;
	float startY = diam + diam + diam;
	float startZ = -0.5f * containerDepth + diam;
	float yshift = sqrt(3.0f) * coarseR;

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)coarseDamWidth; i++)
		{
			for (unsigned int j = 0; j < coarseDamHeight; j++)
			{
				for (unsigned int k = 0; k < coarseDamDepth; k++)
				{
					p_damParticles[i*coarseDamHeight*coarseDamDepth + j*coarseDamDepth + k] = diam*Eigen::Vector3f((float)i, (float)j, (float)k) + Eigen::Vector3f(startX, startY, startZ);
				}
			}
		}
	}

	std::cout << "Number of particles: " << p_damParticles.size() << "\n";

}
void CreateCoarseContainer(std::vector<Vector3f>& p_boundaryParticles)
{
	float x1 = -containerWidth / 2.0f;
	float x2 = containerWidth / 2.0f;
	float y1 = 0.0f;
	float y2 = containerHeight;
	float z1 = -containerDepth / 2.0f;
	float z2 = containerDepth / 2.0f;

	float diam = 2.0f*coarseR;

	containerStart[0] = x1;
	containerStart[1] = y1;
	containerStart[2] = z1;

	containerEnd[0] = x2;
	containerEnd[1] = y2;
	containerEnd[2] = z2;

	// Floor
	AddWall(Vector3f(x1, y1, z1), Vector3f(x2, y1, z2), p_boundaryParticles, coarseR);
	// Top
	AddWall(Vector3f(x1, y2, z1), Vector3f(x2, y2, z2), p_boundaryParticles, coarseR);
	// Left
	AddWall(Vector3f(x1, y1, z1), Vector3f(x1, y2, z2), p_boundaryParticles, coarseR);
	// Right
	AddWall(Vector3f(x2, y1, z1), Vector3f(x2, y2, z2), p_boundaryParticles, coarseR);
	// Back
	AddWall(Vector3f(x1, y1, z1), Vector3f(x2, y2, z1), p_boundaryParticles, coarseR);
	// Front
	AddWall(Vector3f(x1, y1, z2), Vector3f(x2, y2, z2), p_boundaryParticles, coarseR);
}
void AddWall(Vector3f p_min, Vector3f p_max, std::vector<Vector3f>& p_boundaryParticle, float p_particleRadius)
{
	Vector3f diff = p_max - p_min;
	float diameter = 2 * p_particleRadius;

	int countX = (int)(diff[0] / diameter) + 1;
	int countY = (int)(diff[1] / diameter) + 1;
	int countZ = (int)(diff[2] / diameter) + 1;

	unsigned int startIndex = p_boundaryParticle.size();
	p_boundaryParticle.resize(startIndex + countX*countY*countZ);

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < countX; i++)
		{
			for (int j = 0; j < countY; j++)
			{
				for (int k = 0; k < countZ; k++)
				{
					const Vector3f position = p_min + Vector3f(i*diameter, j*diameter, k*diameter);
					p_boundaryParticle[startIndex + i*countY*countZ + j*countZ + k] = position;
				}
			}
		}
	}
}