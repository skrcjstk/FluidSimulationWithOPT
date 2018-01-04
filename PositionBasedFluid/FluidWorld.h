#pragma once
#ifndef __FLUID_WORLD_H__
#define __FLUID_WORLD_H__

#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "FParticle.h"
#include "FluidKernel.h"
#include "PBF.h"

// 0 - PBF
// 1 - IISPH
// 2 - WCSPH
#define FLUID_METHOD 0

using namespace Eigen;

class FluidWorld
{
private:
	PBFWorld* pbfWorld;
	
	std::vector<FParticle*> m_particles;
	std::vector<FParticle*> m_boundaryParticles;
	
	FluidKernel k;

	int  m_numOfParticles;
	int  m_numOfBoundaryParticles;

	float m_particleRadius;
	float m_smoothingLength;
	float m_restDensity;
	float m_timeStep;
	bool  m_useGravity;

	void ComputeDensities();
	void NeighborListUpdate();
	void UpdateTimeStepSizeCFL();

public:
	float m_accTimeIntegration;

	FluidWorld();
	~FluidWorld();

	void Reset();
	void CreateParticles(std::vector<Vector3f>& p_damParticles, std::vector<Vector3f>& p_containerParticles, float p_particleRadius);

	void StepPBF();
	void StepPBFonSub();
	void StepPBFonSub1();
	void StepPBFonSub2();
	void StepPBFonSub1WithTF();
	void StepPBFonSub2WithTF();
	
	FParticle* GetParticle(int p_index)
	{
		return m_particles[p_index];
	}
	std::vector<FParticle*>& GetParticleList()
	{
		return m_particles;
	}
	FParticle* GetBoundaryParticle(int p_index)
	{
		return m_boundaryParticles[p_index];
	}
	std::vector<FParticle*>& GetBoundaryParticleList()
	{
		return m_boundaryParticles;
	}

	int  GetNumOfParticles()
	{
		return m_numOfParticles;
	}
	int GetNumOfBoundaryParticles()
	{
		return m_numOfBoundaryParticles;
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
	void SetTimeStep(float p_timeStep)
	{
		m_timeStep = p_timeStep;
	}

	FluidKernel& GetKernel() { return k; }
	float GetTimeStep() { return m_timeStep; }
	float GetParticleRadius() { return m_particleRadius;  }
	float GetRestDensity() { return m_restDensity; }

};


#endif
