#include "PBF.h"

PBFWorld::PBFWorld(float p_restDensity, float p_viscosity, float p_surfaceTensionThr, float p_surfaceTensionCoeff)
{
	m_restDensity = p_restDensity;
	m_viscosity = p_viscosity;
	m_surfaceTensionThr = p_surfaceTensionThr;
	m_surfaceTensionCoeff = p_surfaceTensionCoeff;
}

void PBFWorld::Reset()
{
	unsigned int numOfParticles = m_particlesLambda.size();
	for (int i = 0; i < (int)numOfParticles; i++)
	{
		m_particlesLambda[i] = 0.0f;
		m_deltaX[i].setZero();
	}
}

void PBFWorld::InitializeSimulationData(int p_numOfParticles)
{
	m_particlesLambda.clear();
	m_particlesLambda.resize(p_numOfParticles);

	m_deltaX.clear();
	m_deltaX.resize(p_numOfParticles);
}

void PBFWorld::SetSmoothingLength(float p_smoothingLength)
{
	m_kernel.SetSmoothingRadius(p_smoothingLength);
}

void PBFWorld::ConstraintProjection(std::vector<FParticle*>& p_particles, std::vector<Vector3f>& p_boundaryParticles, std::vector<float>& p_boundaryPsi, float p_timeStep)
{
	int maxiter = 100;
	int iter = 0;

	float eps = 1.0e-6;

	unsigned int numParticles = p_particles.size();
	float invH = 1.0 / p_timeStep;
	float invH2 = invH*invH;

	float density0 = m_restDensity;
	float maxError = 0.01f;
	float eta = maxError * 0.01 * density0;  // maxError is given in percent

	float avg_density_err = 0.0f;
	while (((avg_density_err > eta) || (iter < 2)) && (iter < maxiter))
	{
		avg_density_err = 0.0f;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)
			for (int i = 0; i < numParticles; i++)
			{
				// computePBFDensity
				p_particles[i]->m_density = p_particles[i]->m_mass * m_kernel.Cubic_Kernel0();

				for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
				{
					unsigned int idx = p_particles[i]->m_neighborList[j];
					Vector3f r = p_particles[i]->m_curPosition - p_particles[idx]->m_curPosition;

					p_particles[i]->m_density += p_particles[idx]->m_mass * m_kernel.Cubic_Kernel(r);
				}

				for (unsigned int j = 0; j < p_particles[i]->m_neighborBoundaryList.size(); j++)
				{
					unsigned int idx = p_particles[i]->m_neighborBoundaryList[j];
					Vector3f r = p_particles[i]->m_curPosition - p_boundaryParticles[idx];
					p_particles[i]->m_density += p_boundaryPsi[idx] * m_kernel.Cubic_Kernel(r);
				}

				float density_err = std::max(p_particles[i]->m_density, density0) - density0;
#pragma omp atomic
				avg_density_err += density_err / (float)numParticles;

				// Evaluate constraint function
				float C = std::max(p_particles[i]->m_density / density0 - 1.0f, 0.0f);

				if (C != 0.0f)
				{
					// Compute gradients dC/dx_j 
					float sum_grad_C2 = 0.0;
					Vector3f gradC_i(0.0f, 0.0f, 0.0f);

					for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
					{
						unsigned int idx = p_particles[i]->m_neighborList[j];
						Vector3f r = p_particles[i]->m_curPosition - p_particles[idx]->m_curPosition;

						Vector3f gradC_j = -p_particles[idx]->m_mass / m_restDensity * m_kernel.Cubic_Kernel_Gradient(r);
						sum_grad_C2 += gradC_j.squaredNorm();
						gradC_i -= gradC_j;
					}

					for (unsigned int j = 0; j < p_particles[i]->m_neighborBoundaryList.size(); j++)
					{
						unsigned int idx = p_particles[i]->m_neighborBoundaryList[j];
						Vector3f r = p_particles[i]->m_curPosition - p_boundaryParticles[idx];

						Vector3f gradC_j = -p_boundaryPsi[idx] / m_restDensity * m_kernel.Cubic_Kernel_Gradient(r);
						sum_grad_C2 += gradC_j.squaredNorm();
						gradC_i -= gradC_j;
					}

					sum_grad_C2 += gradC_i.squaredNorm();

					// Compute lambda
					m_particlesLambda[i] = -C / (sum_grad_C2 + eps);
				}
				else
				{
					m_particlesLambda[i] = 0.0f;
				}
			}

#pragma omp for schedule(static)
			// Compute position correction
			for (int i = 0; i < numParticles; i++)
			{
				Vector3f corr(0.0f, 0.0f, 0.0f);

				for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
				{
					unsigned int idx = p_particles[i]->m_neighborList[j];
					Vector3f r = p_particles[i]->m_curPosition - p_particles[idx]->m_curPosition;

					Vector3f gradC_j = -p_particles[idx]->m_mass / m_restDensity * m_kernel.Cubic_Kernel_Gradient(r);
					corr -= (m_particlesLambda[i] + m_particlesLambda[idx]) * gradC_j;
				}

				for (unsigned int j = 0; j < p_particles[i]->m_neighborBoundaryList.size(); j++)
				{
					unsigned int idx = p_particles[i]->m_neighborBoundaryList[j];
					Vector3f r = p_particles[i]->m_curPosition - p_boundaryParticles[idx];

					Vector3f gradC_j = -p_boundaryPsi[idx] / m_restDensity * m_kernel.Cubic_Kernel_Gradient(r);
					corr -= (m_particlesLambda[i]) * gradC_j;
				}

				m_deltaX[i] = corr;
			}

#pragma omp for schedule(static)
			for (int i = 0; i < numParticles; i++)
			{
				p_particles[i]->m_curPosition += m_deltaX[i];
			}
		}

		iter++;
	}
}

void PBFWorld::ComputeXSPHViscosity(std::vector<FParticle*>& p_particles)
{
	unsigned int numParticles = p_particles.size();

	// Compute viscosity forces (XSPH)
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborList[j];
				Vector3f r = p_particles[i]->m_curPosition - p_particles[idx]->m_curPosition;

				Vector3f velCorr = m_viscosity * (p_particles[idx]->m_mass / p_particles[idx]->m_density)
					* (p_particles[i]->m_velocity - p_particles[idx]->m_velocity) * m_kernel.Cubic_Kernel(r);

				p_particles[i]->m_velocity -= velCorr;
			}
		}
	}
}