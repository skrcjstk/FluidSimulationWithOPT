#include "WCSPH.h"

WCSPHWorld::WCSPHWorld(float p_restDensity, float p_viscosity, float p_surfaceTensionThr, float p_surfaceTensionCoeff, float p_maxError, int p_maxIteration)
{
	m_restDensity = p_restDensity;
	m_viscosity = p_viscosity;
	m_surfaceTensionThr = p_surfaceTensionThr;
	m_surfaceTensionCoeff = p_surfaceTensionCoeff;
	m_maxError = p_maxError;
	m_maxIterations = p_maxIteration;

	m_exponent = 7.0f;
	m_stiffness = 50000.0f;
}

void WCSPHWorld::Reset()
{
	unsigned int numOfParticle = m_pressure.size();
	for (int i = 0; i < (int)numOfParticle; i++)
	{
		m_pressure[i] = 0.0f;
		m_pressureAccel[i].setZero();
	}
}

void WCSPHWorld::SetSmoothingLength(float p_smoothingLength)
{
	m_kernel.SetSmoothingRadius(p_smoothingLength);
}

void WCSPHWorld::InitializeSimulationData(int p_numOfParticles)
{
	m_pressure.clear();
	m_pressure.resize(p_numOfParticles, 0.0);
	m_pressureAccel.clear();
	m_pressureAccel.resize(p_numOfParticles, Vector3f::Zero());
}

void WCSPHWorld::ComputeViscosity(std::vector<FParticle*>& p_particles)
{
	// standard viscosity
	unsigned int numParticles = p_particles.size();
	float h = m_kernel.GetSmoothingRadius();
	float h2 = h*h;

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			float density_i = p_particles[i]->m_density;
			Vector3f& xi = p_particles[i]->m_curPosition;
			Vector3f& vi = p_particles[i]->m_velocity;
			for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborList[j];
				float density_j = p_particles[idx]->m_density;
				Vector3f& xj = p_particles[idx]->m_curPosition;
				Vector3f& vj = p_particles[idx]->m_velocity;

				Vector3f xixj = xi - xj;

				p_particles[i]->m_acceleration += 2.0 * m_viscosity * (p_particles[idx]->m_mass / density_j)
					* (vi - vj) * (xixj.dot(m_kernel.Cubic_Kernel_Gradient(xi - xj))) / (xixj.squaredNorm() + 0.01*h2);
			}
		}
	}
}
void WCSPHWorld::ComputeSurfaceTension()
{
	// skip
}

void WCSPHWorld::ComputePressures(std::vector<FParticle*>& p_particles)
{
	unsigned int numParticles = p_particles.size();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			float &density = p_particles[i]->m_density;
			density = max(density, m_restDensity);
			m_pressure[i] = m_stiffness * (pow(density / m_restDensity, m_exponent) - 1.0);
		}
	}
}
void WCSPHWorld::ComputePressureAccels(std::vector<FParticle*>& p_particles, std::vector<Vector3f>& p_boundaryParticles, std::vector<float>& p_boundaryPsi)
{
	unsigned int numParticles = p_particles.size();

	// Compute pressure forces
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			Vector3f &xi = p_particles[i]->m_curPosition;
			float &density_i = p_particles[i]->m_density;

			Vector3f &ai = m_pressureAccel[i];
			ai.setZero();

			float dpi = m_pressure[i] / (density_i*density_i);
			for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborList[j];
				Vector3f r = xi - p_particles[idx]->m_curPosition;
				
				// Pressure 
				float &density_j = p_particles[idx]->m_density;
				float dpj = m_pressure[idx] / (density_j*density_j);

				ai -= p_particles[idx]->m_mass * (dpi + dpj) * m_kernel.Cubic_Kernel_Gradient(r);
			}
			for (unsigned int j = 0; j < p_particles[i]->m_neighborBoundaryList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborBoundaryList[j];
				Vector3f r = xi - p_boundaryParticles[idx];
				Vector3f a = p_boundaryPsi[idx] * (dpi)* m_kernel.Cubic_Kernel_Gradient(r);
				ai -= a;

				// for moving boundary objects
				//m_model->getForce(particleId.point_set_id, neighborIndex) += m_model->getMass(i) * a;
			}
		}
	}
}
void WCSPHWorld::Integration(std::vector<FParticle*>& p_particles, float p_timeStep)
{
	unsigned int numParticles = p_particles.size();
	float h = p_timeStep;

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static) 
		for (int i = 0; i < (int)numParticles; i++)
		{
			Vector3f &pos = p_particles[i]->m_curPosition;
			Vector3f &vel = p_particles[i]->m_velocity;
			Vector3f &accel = p_particles[i]->m_acceleration;
			accel += m_pressureAccel[i];
			vel += accel * h;
			pos += vel * h;
		}
	}
}