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

void PBFWorld::ConstraintProjection(std::vector<FParticle*>& p_particles, std::vector<FParticle* >& p_boundaryParticles, float p_timeStep)
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
				FParticle* pi = p_particles[i];
				pi->m_density = pi->m_mass * m_kernel.Cubic_Kernel0();

				for (unsigned int j = 0; j < pi->m_neighborList.size(); j++)
				{
					FParticle* pj = pi->m_neighborList[j];
					Vector3f r = pi->m_curPosition - pj->m_curPosition;

					pi->m_density += pj->m_mass * m_kernel.Cubic_Kernel(r);
				}

				float density_err = std::max(pi->m_density, density0) - density0;
#pragma omp atomic
				avg_density_err += density_err / (float)numParticles;

				// Evaluate constraint function
				float C = std::max(pi->m_density / density0 - 1.0f, 0.0f);

				if (C != 0.0f)
				{
					// Compute gradients dC/dx_j 
					float sum_grad_C2 = 0.0;
					Vector3f gradC_i(0.0f, 0.0f, 0.0f);

					for (unsigned int j = 0; j < pi->m_neighborList.size(); j++)
					{
						FParticle* pj = pi->m_neighborList[j];
						Vector3f r = pi->m_curPosition - pj->m_curPosition;

						Vector3f gradC_j = -pj->m_mass / m_restDensity * m_kernel.Cubic_Kernel_Gradient(r);
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
				FParticle* pi = p_particles[i];

				for (unsigned int j = 0; j < pi->m_neighborList.size(); j++)
				{
					FParticle* pj = pi->m_neighborList[j];
					Vector3f r = pi->m_curPosition - pj->m_curPosition;
					Vector3f gradC_j = -pj->m_mass / m_restDensity * m_kernel.Cubic_Kernel_Gradient(r);

					if(pj->m_pid==Fluid)
						corr -= (m_particlesLambda[i] + m_particlesLambda[pj->m_pIdx]) * gradC_j;
					else
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
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		// Compute viscosity forces (XSPH)
		for (int i = 0; i < numParticles; i++)
		{
			FParticle* pi = p_particles[i];
			for (unsigned int j = 0; j < pi->m_neighborList.size(); j++)
			{
				FParticle* pj = pi->m_neighborList[j];
				if (pj->m_pid == Fluid)
				{
					Vector3f r = pi->m_curPosition - pj->m_curPosition;
					Vector3f velCorr = m_viscosity * (pj->m_mass / pj->m_density) * (pi->m_velocity - pj->m_velocity) * m_kernel.Cubic_Kernel(r);

					pi->m_velocity -= velCorr;
				}
			}
		}
	}
}