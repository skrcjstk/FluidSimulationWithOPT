#include "GL/glew.h"
#include "..\FluidSimulationWithOPT\Visualization\MiniGL.h"
#include "..\FluidSimulationWithOPT\Visualization\Selection.h"
#include "GL/glut.h"
#include "..\FluidSimulationWithOPT\Visualization\PrimitiveBuffer.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "PIC.h"
#include "FluidWorld.h"
#include "TimerChrono.h"

using namespace PBD;
using namespace std;
using namespace Eigen;

TimerChrono timer1, timer2, timer3;

float fineR = 0.025f;
float coarseR = 0.05f;
int fineDamWidth = 20;
int fineDamHeight = 20;
int fineDamDepth = 20;
int coarseDamWidth = fineDamWidth / 2;
int coarseDamHeight = fineDamHeight / 2;
int coarseDamDepth = fineDamDepth / 2;
float containerWidth = (coarseDamWidth * 7) * coarseR;
float containerHeight = (coarseDamWidth * 5) * coarseR;
float containerDepth = (coarseDamWidth * 5) * coarseR;
Vector3f containerStart;
Vector3f containerEnd;

bool doPause = true;
int accFrameCount = 0;

PIC* pic;
FluidWorld* world;

int frameLimit = 800;
GLint context_major_version, context_minor_version;
Primitive spherePrimiCoarse, spherePrimiFine, boxPrimi;

float* dataForPICDescriptor;
int desc_width = 5;

void timeStep();
void reset();
void render();
void cleanup();
void buildModel();

void CreateCoarseBreakingDam(std::vector<Vector3f>& p_damParticles);
void CreateCoarseContainer(std::vector<Vector3f>& p_boundaryParticles);
void CreateFineBreakingDam(std::vector<Vector3f>& p_damParticles);
void CreateFineContainer(std::vector<Vector3f>& p_boundaryParticles);
void AddWall(Vector3f p_min, Vector3f p_max, std::vector<Vector3f>& p_boundaryParticle, float p_particleRadius);

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

	timer2.start();
	pic->AssignCells(world);
	pic->Map_P2G(world);
	timer2.end("PIC Update");

	timer3.start();
	pic->GetDescriptorAll(dataForPICDescriptor, desc_width);
	timer3.end("Desc Update");


	accFrameCount++;
	if (accFrameCount > frameLimit)
	{
		doPause = !doPause;
	}
}
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
	for (int i = 0; i < world->GetNumOfParticles(); i++)
	{
		spherePrimiCoarse.renderSphere(world->GetParticle(i)->m_curPosition, fluidColor);
	}

	float boundaryColor[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
	for (int i = 0; i < world->GetNumOfBoundaryParticles(); i++)
	{
		//spherePrimiCoarse.renderSphere(world->GetBoundaryParticle(i)->m_curPosition, boundaryColor);
	}

	// draw grid & arrow
	float FDot[4] = { 0.0f, 0.0f, 0.8f, 1.0f };
	float BDot[4] = { 0.8f, 0.4f, 0.8f, 1.0f };
	float boxColor[4] = { 0.1f, 0.1f, 0.1f, 0.5f };
	Vector3f dxyz = pic->GetDxDyDz();
	float head_len = 0.1f*dxyz[0];

	int ng = pic->GetGridSize();
	for (int g = 0; g < ng; g++)
	{
		Vector3f& pos = pic->GetGridPos(g);
		
		//boxPrimi.renderWireFrameBox(pos, dxyz, boxColor);
		
		//if(pic->GetGid(g) == F)
		//	boxPrimi.renderPoint(pos, FDot, 1.0f);
		
		//else if (pic->GetGid(g) == B)
		//	boxPrimi.renderWireFrameBox(pos, dxyz, boxColor);
		
		Vector3f end = (pos + 0.01f * pic->GetVelocity(g));
		boxPrimi.renderArrow3D(pos, end, head_len);
	}

	/*
	int np = world->GetNumOfParticles();
	int d = desc_width;
	int halfCnt = (int)(desc_width / 2);
	int dataSize = 4 * (d*d*d);
	float* desc = (float*)malloc(sizeof(float) * 4 * desc_width * desc_width * desc_width);

	if (accFrameCount > 1)
	{
		for (int n = 0; n < np; n++)
		{
			Vector3i& ijk = pic->GetAssignResultF(n);
			pic->GetDescriptor(ijk, desc, desc_width);

			Vector3f gridPos = pic->GetGridPos(ijk[0], ijk[1], ijk[2]);

			for (int k = -halfCnt; k <= halfCnt; k++)
				for (int j = -halfCnt; j <= halfCnt; j++)
					for (int i = -halfCnt; i <= halfCnt; i++)
					{
						int idx = 4 * ((k + halfCnt) * (d*d) + (j + halfCnt)*(d)+(i + halfCnt));
						Vector3i neiGrid = ijk + Vector3i(i, j, k);
						Vector3f neiGridPos = pic->GetGridPos(neiGrid[0], neiGrid[1], neiGrid[2]);

						Vector3f vel;
						//result[startidx + idx + 0] = geo[neiIdx];
						vel[0] = desc[idx + 1];
						vel[1] = desc[idx + 2];
						vel[2] = desc[idx + 3];

						Vector3f end = (neiGridPos + 0.01f * vel);
						boxPrimi.renderArrow3D(neiGridPos, end, head_len);
					}
		}
	}
	*/
}

void buildModel()
{
	std::vector<Vector3f> boundaryParticles;
	std::vector<Vector3f> damParticles;
	CreateCoarseBreakingDam(damParticles);
	CreateCoarseContainer(boundaryParticles);
	//CreateFineBreakingDam(damParticles);
	//CreateFineContainer(boundaryParticles);

	world = new FluidWorld();
	world->SetTimeStep(0.005f);
	world->CreateParticles(damParticles, boundaryParticles, coarseR);

	float gDx = 2.0f * coarseR;
	Vector3f gStart = containerStart - Vector3f(5.0f * gDx, 5.0f * gDx, 5.0f * gDx);
	Vector3f gEnd = containerEnd + Vector3f(5.0f*gDx, 5.0f*gDx, 5.0f*gDx);
	Vector3f gSize = gEnd - gStart;	
	pic = new PIC();
	pic->Initialize(world, gStart, gEnd - gStart, Vector3i((int)(gSize[0] / gDx), (int)(gSize[1] / gDx), (int)(gSize[2] / gDx)), 1.0);
	pic->AssignBoundary(world->GetBoundaryParticleList());

	int blockSize = desc_width * desc_width * desc_width;
	dataForPICDescriptor = (float*)malloc(sizeof(float) * 4 * blockSize * damParticles.size());
}
void CreateCoarseBreakingDam(std::vector<Vector3f>& p_damParticles)
{
	std::cout << "Initialize coarse fluid particles\n";
	p_damParticles.resize(coarseDamWidth*coarseDamHeight*coarseDamDepth);

	float diam = 2.0f * coarseR;
	float startX = -0.5f * containerWidth + diam + diam;
	float startY = diam + diam;
	float startZ = -0.5f * containerDepth + diam;

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int k = 0; k < coarseDamDepth; k++)
		{
			for (int j = 0; j < coarseDamHeight; j++)
			{
				for (int i = 0; i < coarseDamWidth; i++)
				{
					p_damParticles[k*coarseDamHeight*coarseDamWidth + j*coarseDamWidth + i] = diam * Vector3f((float)i, (float)j, (float)k) + Vector3f(startX, startY, startZ);
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
	//AddWall(Vector3f(x1, y2, z1), Vector3f(x2, y2, z2), p_boundaryParticles, coarseR);
	// Left
	AddWall(Vector3f(x1, y1, z1), Vector3f(x1, y2, z2), p_boundaryParticles, coarseR);
	// Right
	AddWall(Vector3f(x2, y1, z1), Vector3f(x2, y2, z2), p_boundaryParticles, coarseR);
	// Back
	AddWall(Vector3f(x1, y1, z1), Vector3f(x2, y2, z1), p_boundaryParticles, coarseR);
	// Front
	AddWall(Vector3f(x1, y1, z2), Vector3f(x2, y2, z2), p_boundaryParticles, coarseR);

	std::cout << "Number of Boundary Particles: " << p_boundaryParticles.size() << "\n";
}
void AddWall(Vector3f p_min, Vector3f p_max, std::vector<Vector3f>& p_boundaryParticle, float p_particleRadius)
{
	Vector3f diff = p_max - p_min;
	float diameter = 2 * p_particleRadius;

	int countX = (int)(diff[0] / diameter) + 1;
	int countY = (int)(diff[1] / diameter) + 1;
	int countZ = (int)(diff[2] / diameter) + 1;

	int startIndex = (int)p_boundaryParticle.size();
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
void CreateFineBreakingDam(std::vector<Vector3f>& p_damParticles)
{
	std::cout << "Initialize fine fluid particles\n";
	p_damParticles.resize(fineDamWidth*fineDamHeight*fineDamDepth);

	float diam = 2.0f * fineR;
	float coarseDiam = 2.0f * coarseR;
	float startX = -0.5f * containerWidth + coarseDiam + coarseDiam;
	float startY = coarseDiam + coarseDiam + coarseDiam;
	float startZ = -0.5f * containerDepth + coarseDiam;

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)fineDamWidth; i++)
		{
			for (int j = 0; j < fineDamHeight; j++)
			{
				for (int k = 0; k < fineDamDepth; k++)
				{
					p_damParticles[i*fineDamHeight*fineDamDepth + j*fineDamDepth + k] = diam*Eigen::Vector3f((float)i, (float)j, (float)k) + Eigen::Vector3f(startX, startY, startZ);
				}
			}
		}
	}

	std::cout << "Number of particles: " << p_damParticles.size() << "\n";
}
void CreateFineContainer(std::vector<Vector3f>& p_boundaryParticles)
{
	float x1 = -containerWidth / 2.0f;
	float x2 = containerWidth / 2.0f;
	float y1 = 0.0f;
	float y2 = containerHeight;
	float z1 = -containerDepth / 2.0f;
	float z2 = containerDepth / 2.0f;

	containerStart[0] = x1;
	containerStart[1] = y1;
	containerStart[2] = z1;

	containerEnd[0] = x2;
	containerEnd[1] = y2;
	containerEnd[2] = z2;

	// Floor
	AddWall(Vector3f(x1, y1, z1), Vector3f(x2, y1, z2), p_boundaryParticles, fineR);
	// Top
	//AddWall(Vector3f(x1, y2, z1), Vector3f(x2, y2, z2), p_boundaryParticles, fineR);
	// Left
	AddWall(Vector3f(x1, y1, z1), Vector3f(x1, y2, z2), p_boundaryParticles, fineR);
	// Right
	AddWall(Vector3f(x2, y1, z1), Vector3f(x2, y2, z2), p_boundaryParticles, fineR);
	// Back
	AddWall(Vector3f(x1, y1, z1), Vector3f(x2, y2, z1), p_boundaryParticles, fineR);
	// Front
	AddWall(Vector3f(x1, y1, z2), Vector3f(x2, y2, z2), p_boundaryParticles, fineR);

	std::cout << "Number of Boundary Particles: " << p_boundaryParticles.size() << "\n";
}

void reset() { accFrameCount = 0; }
void cleanup()
{
	timer1.printAvg("Avg Simulation");
	timer2.printAvg("Avg PIC Update");
	timer3.printAvg("Avg Des Update");

	if (context_major_version >= 3)
	{
		spherePrimiCoarse.releaseBuffers();
		spherePrimiFine.releaseBuffers();
		boxPrimi.releaseBuffers();
	}
	pic->clean();
	free(dataForPICDescriptor);
}
void selection(const Eigen::Vector2i &start, const Eigen::Vector2i &end) {}