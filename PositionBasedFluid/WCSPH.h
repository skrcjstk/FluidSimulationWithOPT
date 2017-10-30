#pragma once
#ifndef __WCSPH_H__
#define __WCSPH_H__

#include <stdlib.h>
#include <Eigen/Dense>
#include <vector>
#include "FParticle.h"
#include "FluidKernel.h"

using namespace Eigen;
using namespace std;

class WCSPHWorld
{
private:
	// simulation data
	
	FluidKernel m_kernel;
	float m_restDensity;
	float m_viscosity;
	float m_surfaceTensionThr;
	float m_surfaceTensionCoeff;
	float m_maxError;
	int   m_maxIterations;

public:
	float m_stiffness;
	float m_exponent;
	std::vector<float> m_pressure;
	std::vector<Vector3f> m_pressureAccel;

	WCSPHWorld(float p_restDensity, float p_viscosity, float p_surfaceTensionThr, float p_surfaceTensionCoeff, float p_maxError, int p_maxIteration);
	void Reset();

	void InitializeSimulationData(int p_numOfParticles);
	void SetSmoothingLength(float p_smoothingLength);

	void ComputeViscosity(std::vector<FParticle*>& p_particles);
	void ComputeSurfaceTension();
	void ComputePressures(std::vector<FParticle*>& p_particles);
	void ComputePressureAccels(std::vector<FParticle*>& p_particles, std::vector<Vector3f>& p_boundaryParticles, std::vector<float>& p_boundaryPsi);
	void Integration(std::vector<FParticle*>& p_particles, float p_timeStep);
};

#endif