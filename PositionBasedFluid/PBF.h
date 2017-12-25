#pragma once
#ifndef __PBF_H__
#define __PBF_H__

#include <stdlib.h>
#include <Eigen/Dense>
#include <vector>
#include "FParticle.h"
#include "FluidKernel.h"

using namespace Eigen;
using namespace std;

class PBFWorld
{
private:
	// simulation data
	std::vector<float> m_particlesLambda;
	std::vector<Vector3f> m_deltaX;

	float m_viscosity;
	float m_surfaceTensionThr;
	float m_surfaceTensionCoeff;
	float m_restDensity;
	FluidKernel m_kernel;


public:
	PBFWorld(float p_restDensity, float p_viscosity, float p_surfaceTensionThr, float p_surfaceTensionCoeff);
	void Reset();

	void InitializeSimulationData(int p_numOfParticles);
	void SetSmoothingLength(float p_smoothingLength);
	void ComputeXSPHViscosity(std::vector<FParticle*>& m_particles);
	void ConstraintProjection(std::vector<FParticle*>& p_particles, std::vector<FParticle*>& p_boundaryParticles, float p_timeStep);
};

#endif
