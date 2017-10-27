#pragma once
#ifndef __FLUID_WORLD_H__
#define __FLUID_WORLD_H__

#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "FParticle.h"
#include "FluidKernel.h"
#include "PBF.h"
#include "IISPH.h"

// 0 - PBF
// 1 - IISPH
#define FLUID_METHOD 1

using namespace Eigen;

class FluidWorld
{
private:
	PBFWorld* pbfWorld;
	IISPHWorld* iisphWorld;
	
	std::vector<FParticle*> m_particles;
	std::vector<Vector3f> m_boundaryParticles;
	std::vector<float> m_boundaryPsi;

	FluidKernel k;

	unsigned int  m_numOfParticles;
	unsigned int  m_numOfBoundaryParticles;

	float m_particleRadius;
	float m_smoothingLength;
	float m_restDensity;
	float m_timeStep;
	bool  m_useGravity;

	void ComputeDensities();
	void NeighborListUpdate();
	void UpdateTimeStepSizeCFL();

	bool debugFlag = false;

public:
	
	float m_accTimeIntegration;

	FluidWorld();
	~FluidWorld();

	int  GetFluidMethodNumber();
	void* GetFluidMethod();


	void Reset();
	void CreateParticles(std::vector<Vector3f>& p_damParticles, std::vector<Vector3f>& p_containerParticles, float p_particleRadius);
	void AddFParticle(Vector3f p_position, Vector3f p_velocity);
	void DeleteFParticle(int p_id);
	void DeleteAll();

	void StepPBF();
	void StepPBFonFine();
	
	void StepIISPH();

	void StepIISPHonCoarse1();
	void StepIISPHonCoarse2();
	void StepIISPHonFine();
	
	FParticle* GetParticle(int p_index)
	{
		return m_particles[p_index];
	}
	std::vector<FParticle*>& GetParticleList()
	{
		return m_particles;
	}

	unsigned int  GetNumOfParticles()
	{
		return m_numOfParticles;
	}
	unsigned int GetNumOfBoundaryParticles()
	{
		return m_numOfBoundaryParticles;
	}
	Vector3f GetBoundaryParticlePosition(int p_index)
	{
		return m_boundaryParticles[p_index];
	}

	float GetSmoothingLength()
	{
		return m_smoothingLength;
	}

	void SetSmoothingLength(float p_smoothingLength)
	{
		m_smoothingLength = p_smoothingLength;
	}
	void SetUseGravity(bool p_useGravity)
	{
		m_useGravity = p_useGravity;
	}

	FluidKernel& GetKernel() { return k; }
	double GetTimeStep() { return m_timeStep; }
	float GetParticleRadius() { return m_particleRadius;  }
	float GetRestDensity() { return m_restDensity; }

};


#endif
