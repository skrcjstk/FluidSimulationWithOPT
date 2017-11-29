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

//#define ENV_LOAD

using namespace PBD;
using namespace Eigen;
using namespace std;

void timeStep();
void reset();
void selection(const Eigen::Vector2i &start, const Eigen::Vector2i &end);
void render();
void buildModel_EnvLoad();
void buildModel_BreakingDam();
void cleanup();
void LoadContainerAndFluidDam(string p_path, std::vector<Vector3f>& p_boundaryParticles, std::vector<Vector3f>& p_damParticles, float& p_radius);
void CreateCoarseBreakingDam(std::vector<Vector3f>& p_damParticles);
void CreateFineBreakingDam(std::vector<Vector3f>& p_damParticles);
void CreateCoarseContainer(std::vector<Vector3f>& p_boundaryParticles);
void AddWall(Vector3f p_min, Vector3f p_max, std::vector<Vector3f>& p_boundaryParticle, float p_particleRadius);
void CreateFineContainer(std::vector<Vector3f>& p_boundaryParticles);
void DataSave();

Primitive spherePrimiCoarse, spherePrimiFine, boxPrimi;

FluidWorld* world;
FluidWorld* subWorld;
FlowBoundary fb;

float fineR = 0.025f;
float coarseR = 2 * fineR;
bool doPause = true;

int fineDamWidth = 20;
int fineDamHeight = 20;
int fineDamDepth = 20;
int coarseDamWidth = fineDamWidth / 2;
int coarseDamHeight = fineDamHeight / 2;
int coarseDamDepth = fineDamDepth / 2;

float containerWidth = (coarseDamWidth + 1) * coarseR * 2.0f * 5.0f;
float containerHeight = 1.5f;
float containerDepth = (coarseDamDepth + 1) * coarseR * 2.0f;

Vector3f containerStart;
Vector3f containerEnd;

Vector3f subContainerStart;
Vector3f subContainerEnd;

GLint context_major_version, context_minor_version;

int accFrameCount = 0;
string TDPath = "./PBFC_SD8/SD";
int saveFrameLimit = 1000;

string coarseEnvPath = "./ObstacleScenes/171124/DamBreakModelDragons_coarse.dat";
string fineEnvPath = "./ObstacleScenes/171124/DamBreakModelDragons_fine.dat";

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

#ifndef ENV_LOAD
	buildModel_BreakingDam();
#elif
	buildModel_EnvLoad();
#endif

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
		if (world->GetFluidMethodNumber() == 0) // PBF case
		{
			// PBFC simulation
			// coarse simulation
			world->StepPBF();

			// fine advection and neighbor update
			subWorld->StepPBFonFine1();
			
			// neighbor update between fine and coarse
			fb.NeighborBTWTwoResForPBFC(world, subWorld);

			// update lambda for coarse & solve density and velocity constraints
			fb.SolvePBFCConstaints(world, subWorld);

			// fine density relaxing and update
			subWorld->StepPBFonFine2();

			// Data save
			DataSave();
		}
		else if (world->GetFluidMethodNumber() == 1) // IISPH case
		{
			fb.NeighborSearchBTWTwoRes(world, subWorld);

			world->StepIISPHonCoarse1();
			fb.InterpolateIISPH(world, subWorld);
			subWorld->StepIISPHonFine();
			world->StepIISPHonCoarse2();
			
		}
		else if (world->GetFluidMethodNumber() == 2) // WCSPH case
		{
			

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
			
			
			//if (world->accFrameCount % 10 == -1)
			//{
				//fb.NeighborSearchBTWTwoRes(world, subWorld);

				//subWorld->StepWCSPHonFine1();
				//world->StepWCSPHonCoarse1();
				//fb.InterpolateWCSPH(world, subWorld, false);
				//world->StepWCSPHonCoarse2();
				//subWorld->StepWCSPHonFine2();	
			//}
			//else
			//{
			//	world->StepWCSPH();
			//	subWorld->StepWCSPH();
			//}
			
		}
	}
	if (accFrameCount > saveFrameLimit)
	{
		doPause = !doPause;
	}
	accFrameCount++;
}

void reset()
{
	world->Reset();
	subWorld->Reset();
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

void buildModel_EnvLoad()
{
	// main domain creation
	std::vector<Vector3f> boundaryParticles;
	std::vector<Vector3f> damParticles;
	LoadContainerAndFluidDam(coarseEnvPath, boundaryParticles, damParticles, coarseR);
	
	world = new FluidWorld();
	world->SetTimeStep(0.01f);
	world->CreateParticles(damParticles, boundaryParticles, coarseR);

	// sub domain creation
	std::vector<Vector3f> subDamParticles;
	std::vector<Vector3f> subBoundaryParticles;
	LoadContainerAndFluidDam(fineEnvPath, subBoundaryParticles, subDamParticles, fineR);

	subWorld = new FluidWorld();
	subWorld->SetTimeStep(0.01f);
	subWorld->CreateParticles(subDamParticles, subBoundaryParticles, fineR);

	fb.SetParticleRadius(fineR);
	fb.InitializeDataStructure(world, subWorld);
	printf("coarse: %d, fine : %d\n", world->GetNumOfParticles(), subWorld->GetNumOfParticles());
}

void buildModel_BreakingDam()
{
	// main domain creation
	std::vector<Vector3f> boundaryParticles;
	std::vector<Vector3f> damParticles;
	CreateCoarseBreakingDam(damParticles);
	CreateCoarseContainer(boundaryParticles);
			
	world = new FluidWorld();
	world->SetTimeStep(0.01f);
	world->CreateParticles(damParticles, boundaryParticles, coarseR);

	// sub domain creation
	std::vector<Vector3f> subDamParticles;
	std::vector<Vector3f> subBoundaryParticles;
	CreateFineBreakingDam(subDamParticles);
	CreateFineContainer(subBoundaryParticles);
	
	subWorld = new FluidWorld();
	subWorld->SetTimeStep(0.01f);
	subWorld->CreateParticles(subDamParticles, subBoundaryParticles, fineR);

	// FlowBoundary Setting and create fine ps
	fb.SetParticleRadius(fineR);
	fb.InitializeDataStructure(world, subWorld);
	printf("coarse: %d, fine : %d\n", world->GetNumOfParticles(), subWorld->GetNumOfParticles());
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
void CreateFineBreakingDam(std::vector<Vector3f>& p_damParticles)
{
	std::cout << "Initialize fine fluid particles\n";
	p_damParticles.resize(fineDamWidth*fineDamHeight*fineDamDepth);

	float diam = 2.0f * fineR;
	float coarseDiam = 2.0f * coarseR;
	float startX = -0.5f * containerWidth + coarseDiam + coarseDiam;
	float startY = coarseDiam + coarseDiam + coarseDiam;
	float startZ = -0.5f * containerDepth + coarseDiam;
	float yshift = sqrt(3.0f) * fineR;

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)fineDamWidth; i++)
		{
			for (unsigned int j = 0; j < fineDamHeight; j++)
			{
				for (unsigned int k = 0; k < fineDamDepth; k++)
				{
					p_damParticles[i*fineDamHeight*fineDamDepth + j*fineDamDepth + k] = diam*Eigen::Vector3f((float)i, (float)j, (float)k) + Eigen::Vector3f(startX, startY, startZ);
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
void CreateFineContainer(std::vector<Vector3f>& p_boundaryParticles)
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

void LoadContainerAndFluidDam(string path, std::vector<Vector3f>& p_boundaryParticles, std::vector<Vector3f>& p_damParticles, float& p_radius)
{
	FILE* fpEnv = fopen(path.c_str(), "rb");

	unsigned int intBuf[1];
	float floatBuf[3];

	// radius 
	fread(floatBuf, sizeof(float), 1, fpEnv);
	printf("particle Radius : %f\n", floatBuf[0]);
	p_radius = floatBuf[0];

	// total boundary particles
	fread(intBuf, sizeof(int), 1, fpEnv);
	printf("nBoundaryParticles : %d\n", intBuf[0]);
	

	// nBoundary objects
	fread(intBuf, sizeof(int), 1, fpEnv);
	printf("nBoundary Objects : %d\n", intBuf[0]);
	int nBObjects = intBuf[0];

	for (int i = 0; i < nBObjects; i++)
	{
		fread(intBuf, sizeof(int), 1, fpEnv);
		printf("(%d) nBoundary particles : %d\n", i, intBuf[0]);
		int nBParticles = intBuf[0];

		for (int j = 0; j < nBParticles; j++)
		{
			fread(floatBuf, sizeof(float), 3, fpEnv);
			p_boundaryParticles.push_back(Vector3f(floatBuf[0], floatBuf[1], floatBuf[2]));
		}
	}

	// nFluid particles
	fread(intBuf, sizeof(int), 1, fpEnv);
	printf("nFluid Particles : %d\n", intBuf[0]);
	int nFParticles = intBuf[0];

	for (int i = 0; i < nFParticles; i++)
	{
		fread(floatBuf, sizeof(float), 3, fpEnv);
		p_damParticles.push_back(Vector3f(floatBuf[0], floatBuf[1], floatBuf[2]));
	}
	
	fclose(fpEnv);
}

void DataSave()
{
	std::vector<FParticle*>& fineP = subWorld->GetParticleList();

	string frameIdx = std::to_string(accFrameCount) + ".dat";
	string tgtFileName = TDPath + frameIdx;
	FILE* fp = fopen(tgtFileName.c_str(), "wb");

	float fbuf[3];
	int ibuf[1];

	// num of fine particles 
	ibuf[0] = (int)fineP.size();
	fwrite(ibuf, sizeof(int), 1, fp);

	for (int i = 0; i < fineP.size(); i++)
	{
		// fine's GT deltaP
		Vector3f deltaP = fineP[i]->m_curPosition - fineP[i]->m_tempPosition;
		fbuf[0] = deltaP[0]; fbuf[1] = deltaP[1]; fbuf[2] = deltaP[2];
		fwrite(fbuf, sizeof(float), 3, fp);
		
		// fine's temp deltaP
		deltaP = fineP[i]->m_tempPosition - fineP[i]->m_oldPosition;
		fbuf[0] = deltaP[0]; fbuf[1] = deltaP[1]; fbuf[2] = deltaP[2];
		fwrite(fbuf, sizeof(float), 3, fp);

		// fine's numOfNeighbors of coarse fluid particle
		ibuf[0] = fb.m_trainData[i].size();
		fwrite(ibuf, sizeof(int), 1, fp);
		for (int j = 0; j < ibuf[0]; j++)
		{
			Vector3f& RVec = fb.m_trainData[i][j].RVec;
			Vector3f& RVel = fb.m_trainData[i][j].RVel;
			float& weight = fb.m_trainData[i][j].weight;

			// weight
			fbuf[0] = weight;
			fwrite(fbuf, sizeof(float), 1, fp);

			// r
			fbuf[0] = RVec[0]; fbuf[1] = RVec[1]; fbuf[2] = RVec[2];
			fwrite(fbuf, sizeof(float), 3, fp);

			// deltaP : coarse's curPos - coarse's oldPos
			fbuf[0] = RVel[0]; fbuf[1] = RVel[1]; fbuf[2] = RVel[2];
			fwrite(fbuf, sizeof(float), 3, fp);
		}
		

		// fine's numOfNeighbors of coarse boundary particle
		ibuf[0] = fb.m_trainDataForBoundary[i].size();
		fwrite(ibuf, sizeof(int), 1, fp);

		for (int j = 0; j < ibuf[0]; j++)
		{
			Vector3f& RVec = fb.m_trainDataForBoundary[i][j].RVec;
			Vector3f& RVel = fb.m_trainDataForBoundary[i][j].RVel;
			float& weight = fb.m_trainDataForBoundary[i][j].weight;

			// weight
			fbuf[0] = weight;
			fwrite(fbuf, sizeof(float), 1, fp);

			// r
			fbuf[0] = RVec[0]; fbuf[1] = RVec[1]; fbuf[2] = RVec[2];
			fwrite(fbuf, sizeof(float), 3, fp);

			// RVel : coarse's Vel - fine's Vel
			fbuf[0] = RVel[0]; fbuf[1] = RVel[1]; fbuf[2] = RVel[2];
			fwrite(fbuf, sizeof(float), 3, fp);
		}


		// fine's numOfNeighbors of fine fluid particle
		ibuf[0] = fb.m_trainDataForFineNeighbor[i].size();
		fwrite(ibuf, sizeof(int), 1, fp);

		for (int j = 0; j < ibuf[0]; j++)
		{
			Vector3f RVec = fb.m_trainDataForFineNeighbor[i][j].RVec;
			Vector3f RVel = fb.m_trainDataForFineNeighbor[i][j].RVel;
			float weight = fb.m_trainDataForFineNeighbor[i][j].weight;

			// weight
			fbuf[0] = weight;
			fwrite(fbuf, sizeof(float), 1, fp);

			// r
			fbuf[0] = RVec[0]; fbuf[1] = RVec[1]; fbuf[2] = RVec[2];
			fwrite(fbuf, sizeof(float), 3, fp);

			// deltaP : fine's tempPos - coarse's oldPos
			fbuf[0] = RVel[0]; fbuf[1] = RVel[1]; fbuf[2] = RVel[2];
			fwrite(fbuf, sizeof(float), 3, fp);
		}
		
	}

	fclose(fp);
}