#pragma once
#ifndef __IISPH_H__
#define __IISPH_H__

#include <stdlib.h>
#include <Eigen/Dense>
#include <vector>
#include "FParticle.h"
#include "FluidKernel.h"

using namespace Eigen;
using namespace std;

class IISPHWorld
{
private:
	// simulation data
	std::vector<float> m_aii;
	std::vector<Vector3f> m_dii;
	std::vector<Vector3f> m_dij_pj;
	std::vector<float> m_density_adv;
	std::vector<float> m_pressure;
	std::vector<float> m_lastPressure;
	std::vector<Vector3f> m_pressureAccel;

	FluidKernel m_kernel;
	float m_restDensity;
	float m_viscosity;
	float m_surfaceTensionThr;
	float m_surfaceTensionCoeff;
	float m_maxError;
	int   m_maxIterations;

public:
	IISPHWorld(float p_restDensity, float p_viscosity, float p_surfaceTensionThr, float p_surfaceTensionCoeff, float p_maxError, int p_maxIteration);
	void Reset();

	void InitializeSimulationData(int p_numOfParticles);
	void SetSmoothingLength(float p_smoothingLength);

	void ComputeViscosity(std::vector<FParticle*>& p_particles);
	void ComputeSurfaceTension();
	void VelocityAdvection(std::vector<FParticle*>& p_particles, float p_timeStep);
	void PredictAdvection(std::vector<FParticle*>& p_particles, std::vector<Vector3f>& p_boundaryParticles, std::vector<float>& p_boundaryPsi, float p_timeStep);
	void PressureSolve(std::vector<FParticle*>& p_particles, std::vector<Vector3f>& p_boundaryParticles, std::vector<float>& p_boundaryPsi, float p_timeStep);
	void Integration(std::vector<FParticle*>& p_particles, std::vector<Vector3f>& p_boundaryParticles, std::vector<float>& p_boundaryPsi, float p_timeStep);
};

#endif