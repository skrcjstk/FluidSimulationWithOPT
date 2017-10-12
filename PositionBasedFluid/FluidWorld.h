#pragma once
#ifndef __FLUID_WORLD_H__
#define __FLUID_WORLD_H__

#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include "FParticle.h"
#include "FluidKernel.h"

using namespace Eigen;

class FluidWorld
{
private:
	std::vector<FParticle*> m_particles;
	std::vector<float> m_particlesLambda;
	std::vector<Vector3f> m_boundaryParticles;
	std::vector<float> m_boundaryPsi;

	FluidKernel k;

	unsigned int  m_numOfParticles;
	unsigned int  m_numOfBoundaryParticles;

	float m_particleRadius;
	float m_smoothingLength;
	float m_viscosity;
	float m_surfaceTensionThr;
	float m_surfaceTensionCoeff;
	float m_restDensity;

	float m_timeStep;
	bool  m_useGravity;

	void ComputeXSPHViscosity();
	void NeighborListUpdate();
	void UpdateTimeStepSizeCFL();
	void ConstraintProjection();

	bool debugFlag = false;

public:

	float m_accTimeIntegration;

	FluidWorld();
	~FluidWorld();

	void Reset();
	void CreateParticles(std::vector<Vector3f>& p_damParticles, std::vector<Vector3f>& p_containerParticles, float p_particleRadius);
	void AddFParticle(Vector3f p_position, Vector3f p_velocity);
	void DeleteFParticle(int p_id);
	void DeleteAll();
	void StepPBF(); 
	
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

};


#endif
