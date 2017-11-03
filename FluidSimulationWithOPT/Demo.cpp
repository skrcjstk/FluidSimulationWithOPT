#include "GL/glew.h"
#include "Visualization\MiniGL.h"
#include "Visualization\Selection.h"
#include "GL/glut.h"
#include "PrimitiveBuffer.h"
#include "FluidWorld.h"
#include "FlowBoundary.h"
#include <Eigen/Dense>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace PBD;
using namespace Eigen;
using namespace std;

void timeStep();
void reset();
void selection(const Eigen::Vector2i &start, const Eigen::Vector2i &end);
void render();
void buildModel();
void cleanup();

Primitive spherePrimiCoarse, spherePrimiFine;
Primitive boxPrimi;

void CreateBreakingDam(std::vector<Vector3f>& p_damParticles);
void CreateContainer(std::vector<Vector3f>& p_boundaryParticles);
void AddWall(Vector3f p_min, Vector3f p_max, std::vector<Vector3f>& p_boundaryParticle, float p_particleRadius);

void SubCreateContainer(std::vector<Vector3f>& p_boundaryParticles);

FluidWorld* world;
FluidWorld* subWorld;
FlowBoundary fb;

const float coarseR = 0.05f;
const float fineR = coarseR * 0.5f;
bool doPause = true;

int damWidth = 5.0f;
int damHeight = 5.0f;
int damDepth = 5.0f;

float containerWidth = (damWidth + 1)*coarseR*2.0f * 5.0f;
float containerHeight = 1.5f;
float containerDepth = (damDepth + 1)*coarseR*2.0f;
Vector3f containerStart;
Vector3f containerEnd;

Vector3f subContainerStart;
Vector3f subContainerEnd;

GLint context_major_version, context_minor_version;

int accFrameCount = 0;

int main(int argc, char** argv)
{
	// OpenGL	
	MiniGL::init(argc, argv, 1024, 768, 0, 0, "Fluid demo");
	MiniGL::initLights();
	MiniGL::setClientIdleFunc(50, timeStep);
	MiniGL::setKeyFunc(0, 'r', reset);
	MiniGL::setSelectionFunc(selection);

	MiniGL::getOpenGLVersion(context_major_version, context_minor_version);

	MiniGL::setClientSceneFunc(render);
	MiniGL::setViewport(40.0f, 0.1f, 500.0f, Vector3f(0.0, 2.0, 8.0), Vector3f(0.0, 2.0, 0.0));

	TwAddVarRW(MiniGL::getTweakBar(), "Pause", TW_TYPE_BOOLCPP, &doPause, " label='Pause' group=Simulation key=SPACE ");

	buildModel();

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

	// Simulation code
	for (unsigned int i = 0; i < 1; i++)
	{
		//fb.NeighborSearchBTWTwoRes(world, subWorld);
		
		if (world->GetFluidMethodNumber() == 0) // PBF case
		{
			fb.InterpolateVelocity(world, subWorld);
			world->StepPBF();
			subWorld->StepPBFonFine();
		}
		else if (world->GetFluidMethodNumber() == 1) // IISPH case
		{
			world->StepIISPHonCoarse1();
			fb.InterpolateIISPH(world, subWorld);
			subWorld->StepIISPHonFine();
			world->StepIISPHonCoarse2();
			
		}
		else if (world->GetFluidMethodNumber() == 2) // WCSPH case
		{
			accFrameCount++;

			//if (accFrameCount % 10 == 0)
			{
				fb.NeighborSearchBTWTwoRes2(world, subWorld);
				world->StepWCSPHonFine1();
				subWorld->StepWCSPHonCoarse1();
				fb.InterpolateWCSPH2(world, subWorld, false);
				subWorld->StepWCSPHonCoarse2();
				world->StepWCSPHonFine2();
			}
			//else
			//{
			//	world->StepWCSPH();
			//	subWorld->StepWCSPH();
			//}
			
			/*
			if (world->accFrameCount % 10 == -1)
			{
				fb.NeighborSearchBTWTwoRes(world, subWorld);

				subWorld->StepWCSPHonFine1();
				world->StepWCSPHonCoarse1();
				fb.InterpolateWCSPH(world, subWorld, false);
				world->StepWCSPHonCoarse2();
				subWorld->StepWCSPHonFine2();
				
			}
			else
			{
				world->StepWCSPH();
				subWorld->StepWCSPH();
			}
			*/
		}
	}
	doPause = !doPause;
}

void reset()
{
	world->Reset();
	subWorld->DeleteAll();
	fb.CreateFinePs(world, subWorld);
}

void selection(const Eigen::Vector2i &start, const Eigen::Vector2i &end) {}

void render()
{
	MiniGL::coordinateSystem();
	MiniGL::drawTime(world->m_accTimeIntegration);

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

	double vmax = 0.4*2.0*world->GetSmoothingLength() / world->GetTimeStep();
	double vmin = 0.0;

	// drawing fluid particles
	float fluidColor[4] = { 0.0f, 0.7f, 0.7f, 0.2f };
	for (int i = 0; i < world->GetNumOfParticles(); i++)
	{
		spherePrimiCoarse.renderSphere(world->GetParticle(i)->m_curPosition, fluidColor);
	}

	// drawing boundary particles
	for (int i = 0; i < world->GetNumOfBoundaryParticles(); i++)
	{
		//spherePrimiCoarse.renderSphere(world->GetBoundaryParticlePosition(i), surfaceColor);
	}

	Vector3f translation(0.0f, 1.5f, 0.0f);

	// drawing inFlowCell
	float halfGridSize = fb.GetGridSize() / 2.0f;
	float cellColor[4] = { 0.2f, 0.2f, 0.8f, 0.3f };
	for (int i=0; i<fb.GetBoundarySize(); i++)
	{
		Vector3f& cellPos = fb.GetFlowBoundary()[i].m_centerPosition;
		//boxPrimi.renderBox(cellPos + translation, Vector3f(halfGridSize, halfGridSize, halfGridSize), cellColor);
	}

	// drawing subBoundary particles
	float subContainerColor[4] = { 0.2f, 0.0f, 0.0f, 0.1f };
	for (int i = 0; i < subWorld->GetNumOfBoundaryParticles(); i++)
	{
		//spherePrimiFine.renderSphere(subWorld->GetBoundaryParticlePosition(i) + translation, subContainerColor);
	}

	// drawing subWorld Fparticles
	float subFluidColor[4] = { 0.2f, 0.3f, 0.9f, 0.8f };
	
	for (int i = 0; i < subWorld->GetNumOfParticles(); i++)
	{
		spherePrimiFine.renderSphere(subWorld->GetParticle(i)->m_curPosition + translation, subFluidColor);
	}
}

void buildModel()
{
	// main domain creation
	std::vector<Vector3f> boundaryParticles;
	std::vector<Vector3f> damParticles;
	
	CreateContainer(boundaryParticles);
	CreateBreakingDam(damParticles);
	
	world = new FluidWorld();
	world->CreateParticles(damParticles, boundaryParticles, coarseR);
	
	// sub domain creation
	std::vector<Vector3f> subDamParticles;
	std::vector<Vector3f> subBoundaryParticles;
	
	SubCreateContainer(subBoundaryParticles);

	subWorld = new FluidWorld();
	subWorld->CreateParticles(subDamParticles, subBoundaryParticles, fineR);

	fb.CreateFlowBoundary(containerStart, containerEnd, fineR);
	fb.SearchBoundaryNeighbor(subBoundaryParticles, fineR);
	fb.SetFluidThreshold(0.35f);

	fb.CreateFinePs(world, subWorld);
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

void CreateBreakingDam(std::vector<Vector3f>& p_damParticles)
{
	std::cout << "Initialize fluid particles\n";
	const float diam = 2.0f*coarseR;
	const float startX = -0.5f*containerWidth + diam + diam;
	const float startY = diam + diam + diam;
	const float startZ = -0.5f*containerDepth + diam;
	const float yshift = sqrt(3.0f) * coarseR;

	p_damParticles.resize(damWidth*damHeight*damDepth);

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)damWidth; i++)
		{
			for (unsigned int j = 0; j < damHeight; j++)
			{
				for (unsigned int k = 0; k < damDepth; k++)
				{
					p_damParticles[i*damHeight*damDepth + j*damDepth + k] = diam*Eigen::Vector3f((float)i, (float)j, (float)k) + Eigen::Vector3f(startX, startY, startZ);
				}
			}
		}
	}

	std::cout << "Number of particles: " << damWidth*damHeight*damDepth << "\n";
}
void CreateContainer(std::vector<Vector3f>& p_boundaryParticles)
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

	unsigned int countX = (unsigned int)(diff[0] / diameter) + 1;
	unsigned int countY = (unsigned int)(diff[1] / diameter) + 1;
	unsigned int countZ = (unsigned int)(diff[2] / diameter) + 1;

	unsigned int startIndex = p_boundaryParticle.size();
	p_boundaryParticle.resize(startIndex + countX*countY*countZ);

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (unsigned int i = 0; i < countX; i++)
		{
			for (unsigned int j = 0; j < countY; j++)
			{
				for (unsigned int k = 0; k < countZ; k++)
				{
					const Vector3f position = p_min + Vector3f(i*diameter, j*diameter, k*diameter);
					p_boundaryParticle[startIndex + i*countY*countZ + j*countZ + k] = position;
				}
			}
		}
	}
}

void SubCreateContainer(std::vector<Vector3f>& p_boundaryParticles)
{
	float x1 = -containerWidth / 2.0f;
	float x2 = containerWidth / 2.0f;
	float y1 = 0.0f;
	float y2 = containerHeight;
	float z1 = -containerDepth / 2.0f;
	float z2 = containerDepth / 2.0f;

	subContainerStart[0] = x1;
	subContainerStart[1] = y1;
	subContainerStart[2] = z1;

	subContainerEnd[0] = x2;
	subContainerEnd[1] = y2;
	subContainerEnd[2] = z2;
	
	// Floor
	AddWall(Vector3f(x1, y1, z1), Vector3f(x2, y1, z2), p_boundaryParticles, fineR);
	// Top
	AddWall(Vector3f(x1, y2, z1), Vector3f(x2, y2, z2), p_boundaryParticles, fineR);
	// Left
	AddWall(Vector3f(x1, y1, z1), Vector3f(x1, y2, z2), p_boundaryParticles, fineR);
	// Right
	AddWall(Vector3f(x2, y1, z1), Vector3f(x2, y2, z2), p_boundaryParticles, fineR);
	// Back
	AddWall(Vector3f(x1, y1, z1), Vector3f(x2, y2, z1), p_boundaryParticles, fineR);
	// Front
	AddWall(Vector3f(x1, y1, z2), Vector3f(x2, y2, z2), p_boundaryParticles, fineR);
}
