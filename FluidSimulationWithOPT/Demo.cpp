#include "GL/glew.h"
#include "Visualization\MiniGL.h"
#include "Visualization\Selection.h"
#include "GL/glut.h"
#include "PrimitiveBuffer.h"
#include "FluidWorld.h"
#include "FlowBoundary.h"
#include "PBFControl.h"
#include "TimerChrono.h"
#include <Eigen/Dense>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "..\APICTest\APICSim.h"
#include <ctime>

//#define ENV_LOAD

using namespace PBD;
using namespace Eigen;
using namespace std;

string positionDataPath = "./PBFCDataScene/";

string TDPath = "./PBFC_SD11_PIC/SD";
int saveFrameLimit = 1600;

string coarseEnvPath = "./ObstacleScenes/171124/DamBreakModelDragons_coarse.dat";
string fineEnvPath = "./ObstacleScenes/171124/DamBreakModelDragons_fine.dat";

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
void PICTrainingDataSave();
void PositionDataSave();

Primitive spherePrimiCoarse, spherePrimiFine, boxPrimi;
TimerChrono timer;

FluidWorld* world;
FluidWorld* subWorld;
PBFControl pbfc;
APICSim picForCoarse, picForFine;
float *descForCoarse, *descForFine, *gtForFine;
int descWidthForC=5, descWidthForF=3;
int sampleCount = 6000;

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
float containerHeight = 3.0f;
float containerDepth = (coarseDamDepth + 1) * coarseR * 2.0f;

Vector3f containerStart;
Vector3f containerEnd;
Vector3f subContainerStart;
Vector3f subContainerEnd;

GLint context_major_version, context_minor_version;

int accFrameCount = 0;

int main(int argc, char** argv)
{
	std::srand(std::time(nullptr));

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
#else
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
		// PBFC simulation
		//float deltaT = subWorld->GetTimeStep();
		//world->SetTimeStep(deltaT);
		world->StepPBF();

		// fine advection and neighbor update
		subWorld->StepPBFonSub1();
			
		picForCoarse.AssignCells(world);
		picForCoarse.Map_P2G(world);
		picForFine.AssignCells(subWorld);
		picForFine.Map_P2G(subWorld);

		// neighbor update between fine and coarse
		pbfc.NeighborBTWTwoResForPBFC(world, subWorld);
		//pbfc.UpdateTrainingDataForMain(world, subWorld);
		//pbfc.UpdateTrainingDataForSub(subWorld);

		// update lambda for coarse & solve density and velocity constraints
		pbfc.SolvePBFCConstaints(world, subWorld);

		// fine density relaxing and update
		subWorld->StepPBFonSub2();

		// Data save
		PICTrainingDataSave();
		//DataSave();
		
	}
	if (accFrameCount > saveFrameLimit)
	{
		doPause = !doPause;
	}
	//PositionDataSave();
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
	//float halfGridSize = fb.GetGridSize() / 2.0f;
	//float cellColor[4] = { 0.2f, 0.2f, 0.8f, 0.3f };
	//for (int i=0; i<fb.GetBoundarySize(); i++)
	//{
	//	Vector3f& cellPos = fb.GetFlowBoundary()[i].m_centerPosition;
		//boxPrimi.renderBox(cellPos + translation, Vector3f(halfGridSize, halfGridSize, halfGridSize), cellColor);
	//}

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
		Vector3f pos = subWorld->GetParticle(i)->m_curPosition + translation;
		spherePrimiFine.renderSphere(pos, subFluidColor);
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

	pbfc.Initialize(world, subWorld);

	//fb.SetParticleRadius(fineR);
	//fb.InitializeDataStructure(world, subWorld);
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
	world->SetTimeStep(0.005f);
	world->CreateParticles(damParticles, boundaryParticles, coarseR);

	// sub domain creation
	std::vector<Vector3f> subDamParticles;
	std::vector<Vector3f> subBoundaryParticles;
	CreateFineBreakingDam(subDamParticles);
	CreateFineContainer(subBoundaryParticles);
	
	subWorld = new FluidWorld();
	subWorld->SetTimeStep(0.005f);
	subWorld->CreateParticles(subDamParticles, subBoundaryParticles, fineR);

	// FlowBoundary Setting and create fine ps
	pbfc.Initialize(world, subWorld);
	printf("coarse: %d, fine : %d\n", world->GetNumOfParticles(), subWorld->GetNumOfParticles());

	// PIC for coarse grid
	Vector3f bSize = containerEnd - containerStart;
	picForCoarse.Initialize(world, containerStart, bSize, Vector3i((int)(bSize[0] * 10), (int)(bSize[1] * 10), (int)(bSize[2] * 10)), 1.0);
	picForCoarse.AssignBoundary(world->GetBoundaryParticleList());

	// PIC for fine grid
	picForFine.Initialize(subWorld, containerStart, bSize, Vector3i((int)(bSize[0] * 10), (int)(bSize[1] * 10), (int)(bSize[2] * 10)), 1.0);

	descForCoarse = (float*)malloc(sizeof(float) * 4 * descWidthForC * descWidthForC * descWidthForC * sampleCount);
	descForFine = (float*)malloc(sizeof(float) * 4 * descWidthForF * descWidthForF * descWidthForF * sampleCount);
	gtForFine = (float*)malloc(sizeof(float) * 3 * sampleCount);

}
void cleanup()
{
	if (context_major_version >= 3)
	{
		spherePrimiCoarse.releaseBuffers();
		spherePrimiFine.releaseBuffers();
		boxPrimi.releaseBuffers();
	}
	free(descForCoarse);
	free(descForFine);
	free(gtForFine);
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
		Vector3f deltaP = fineP[i]->m_curPosition - fineP[i]->m_tempPosition; // for SD9M1
		//Vector3f deltaP = fineP[i]->m_curPosition - (fineP[i]->m_tempPosition + pbfc.m_deltaPWithControl[i]); // for SD9M2
		fbuf[0] = deltaP[0]; fbuf[1] = deltaP[1]; fbuf[2] = deltaP[2];
		fwrite(fbuf, sizeof(float), 3, fp);

		Vector3f tempDeltaP = fineP[i]->m_tempPosition - fineP[i]->m_oldPosition;
		fbuf[0] = tempDeltaP[0]; fbuf[1] = tempDeltaP[1]; fbuf[2] = tempDeltaP[2];
		fwrite(fbuf, sizeof(float), 3, fp);
		
		// fine's deltaPWithControl
		//fbuf[0] = pbfc.m_deltaPWithControl[i][0];
		//fbuf[1] = pbfc.m_deltaPWithControl[i][1];
		//fbuf[2] = pbfc.m_deltaPWithControl[i][2];
		//fwrite(fbuf, sizeof(float), 3, fp);

		// fine's numOfNeighbors of coarse fluid particle
		ibuf[0] = pbfc.m_tDataForMain[i].size();
		fwrite(ibuf, sizeof(int), 1, fp);
		for (int j = 0; j < ibuf[0]; j++)
		{
			float& mass = pbfc.m_tDataForMain[i][j].mass;
			float& kWeight = pbfc.m_tDataForMain[i][j].kWeight;
			Vector3f& RVec = pbfc.m_tDataForMain[i][j].RVec;
			Vector3f& kGrad = pbfc.m_tDataForMain[i][j].kGrad;
			Vector3f& dPos = pbfc.m_tDataForMain[i][j].dPos;
			// mass
			fbuf[0] = mass;
			fwrite(fbuf, sizeof(float), 1, fp);

			// kWeight
			fbuf[0] = kWeight;
			fwrite(fbuf, sizeof(float), 1, fp);

			// kGrad
			fbuf[0] = kGrad[0]; fbuf[1] = kGrad[1]; fbuf[2] = kGrad[2];
			fwrite(fbuf, sizeof(float), 3, fp);
			
			// RVec
			fbuf[0] = RVec[0]; fbuf[1] = RVec[1]; fbuf[2] = RVec[2];
			fwrite(fbuf, sizeof(float), 3, fp);
			
			// dPos
			fbuf[0] = dPos[0]; fbuf[1] = dPos[1]; fbuf[2] = dPos[2];
			fwrite(fbuf, sizeof(float), 3, fp);
		}
		
		// fine's numOfNeighbors of fine fluid particle
		ibuf[0] = pbfc.m_tDataForSub[i].size();
		fwrite(ibuf, sizeof(int), 1, fp);
		for (int j = 0; j < ibuf[0]; j++)
		{
			float& mass = pbfc.m_tDataForSub[i][j].mass;
			float& kWeight = pbfc.m_tDataForSub[i][j].kWeight;
			Vector3f& RVec = pbfc.m_tDataForSub[i][j].RVec;
			Vector3f& kGrad = pbfc.m_tDataForSub[i][j].kGrad;
			Vector3f& dPos = pbfc.m_tDataForSub[i][j].dPos;

			// mass
			fbuf[0] = mass;
			fwrite(fbuf, sizeof(float), 1, fp);
			
			// kWeight
			fbuf[0] = kWeight;
			fwrite(fbuf, sizeof(float), 1, fp);

			// kGrad
			fbuf[0] = kGrad[0]; fbuf[1] = kGrad[1]; fbuf[2] = kGrad[2];
			fwrite(fbuf, sizeof(float), 3, fp);

			// RVec
			fbuf[0] = RVec[0]; fbuf[1] = RVec[1]; fbuf[2] = RVec[2];
			fwrite(fbuf, sizeof(float), 3, fp);

			// dPos
			fbuf[0] = dPos[0]; fbuf[1] = dPos[1]; fbuf[2] = dPos[2];
			fwrite(fbuf, sizeof(float), 3, fp);
		}
	}

	fclose(fp);
}
void PICTrainingDataSave()
{
	std::vector<FParticle*>& fineP = subWorld->GetParticleList();

	string picCFile = TDPath + "forC_" + std::to_string(accFrameCount) + ".dat";
	string picFFile = TDPath + "forF_" + std::to_string(accFrameCount) + ".dat";
	string gtFile = TDPath + "forGT_" + std::to_string(accFrameCount) + ".dat";
	FILE* fpForC = fopen(picCFile.c_str(), "wb");
	FILE* fpForF = fopen(picFFile.c_str(), "wb");
	FILE* fpForGT = fopen(gtFile.c_str(), "wb");

	float fbuf[3];
	int ibuf[1];
	int np = subWorld->GetNumOfParticles();

	int success = 0;
	int sizeForC = 4 * descWidthForC * descWidthForC * descWidthForC;
	int sizeForF = 4 * descWidthForF * descWidthForF * descWidthForF;

	while(success<sampleCount)
	{
		int rnd_idx = std::rand() % np;
		Vector3i& aRes = picForFine.GetAssignResultF(rnd_idx);
		if (aRes[0] != -1 && aRes[1] != -1 && aRes[2] != -1)
		{
			picForCoarse.GetAPICDescriptor(aRes, descForCoarse + success*sizeForC, descWidthForC);
			picForFine.GetAPICDescriptor(aRes, descForFine + success*sizeForF, descWidthForF);
			Vector3f deltaP = fineP[rnd_idx]->m_curPosition - fineP[rnd_idx]->m_tempPosition;

			gtForFine[3 * success + 0] = deltaP[0];
			gtForFine[3 * success + 1] = deltaP[1];
			gtForFine[3 * success + 2] = deltaP[2];

			success += 1;
		}
	}

	fwrite(descForCoarse, sizeof(float), sizeForC * sampleCount, fpForC);
	fwrite(descForFine, sizeof(float), sizeForF * sampleCount, fpForF);
	fwrite(gtForFine, sizeof(float), 3 * sampleCount, fpForGT);
	
	fclose(fpForC);
	fclose(fpForF);
	fclose(fpForGT);
	
	printf("(%d)frame sampled and saved(%d)\n", accFrameCount, success);
}

void PositionDataSave()
{
	int ibuf[1];
	float fbuf[3];

	string filename = positionDataPath + "MainWorld" + std::to_string(accFrameCount) + ".dat";
	FILE* fp = fopen(filename.c_str(), "wb");

	ibuf[0] = world->GetNumOfParticles();
	fwrite(ibuf, sizeof(int), 1, fp);
	for (int i = 0; i < ibuf[0]; i++)
	{
		Vector3f& curPos = world->GetParticle(i)->m_curPosition;
		fbuf[0] = curPos[0];	fbuf[1] = curPos[1];	fbuf[2] = curPos[2];
		//printf("%d's pos %f, %f, %f\n", i, fbuf[0], fbuf[1], fbuf[2]);
		fwrite(fbuf, sizeof(float), 3, fp);
	}
	fclose(fp);

	filename = positionDataPath + "SubWorld" + std::to_string(accFrameCount) + ".dat";
	fp = fopen(filename.c_str(), "wb");

	ibuf[0] = subWorld->GetNumOfParticles();
	fwrite(ibuf, sizeof(int), 1, fp);
	for (int i = 0; i < ibuf[0]; i++)
	{
		Vector3f& curPos = subWorld->GetParticle(i)->m_curPosition;
		fbuf[0] = curPos[0];	fbuf[1] = curPos[1];	fbuf[2] = curPos[2];
		fwrite(fbuf, sizeof(float), 3, fp);
	}
	fclose(fp);
	printf("%d frame's positionData were saved.\n", accFrameCount);
}