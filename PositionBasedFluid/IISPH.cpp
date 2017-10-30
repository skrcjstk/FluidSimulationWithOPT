#include "IISPH.h"

IISPHWorld::IISPHWorld(float p_restDensity, float p_viscosity, float p_surfaceTensionThr, float p_surfaceTensionCoeff, float p_maxError, int p_maxIteration)
{
	m_restDensity = p_restDensity;
	m_viscosity = p_viscosity;
	m_surfaceTensionThr = p_surfaceTensionThr;
	m_surfaceTensionCoeff = p_surfaceTensionCoeff;
	m_maxError = p_maxError;
	m_maxIterations = p_maxIteration;
}

void IISPHWorld::Reset()
{
	unsigned int numOfParticle = m_aii.size();
	for (int i = 0; i < (int)numOfParticle; i++)
	{
		m_aii[i] = 0.0f;
		m_dii[i].setZero();
		m_dij_pj[i].setZero();
		m_density_adv[i] = 0.0f;
		m_pressure[i] = 0.0f; 
		m_lastPressure[i] = 0.0f;
		m_pressureAccel[i].setZero();
	}
}

void IISPHWorld::SetSmoothingLength(float p_smoothingLength)
{
	m_kernel.SetSmoothingRadius(p_smoothingLength);
}

void IISPHWorld::InitializeSimulationData(int p_numOfParticles)
{
	m_aii.clear();
	m_dii.clear();
	m_dij_pj.clear();
	m_density_adv.clear();
	m_pressure.clear();
	m_lastPressure.clear();
	m_pressureAccel.clear();

	m_aii.resize(p_numOfParticles, 0.0);
	m_dii.resize(p_numOfParticles, Vector3f::Zero());
	m_dij_pj.resize(p_numOfParticles, Vector3f::Zero());
	m_density_adv.resize(p_numOfParticles, 0.0);
	m_pressure.resize(p_numOfParticles, 0.0);
	m_lastPressure.resize(p_numOfParticles, 0.0);
	m_pressureAccel.resize(p_numOfParticles, Vector3f::Zero());
}

void IISPHWorld::ComputeViscosity(std::vector<FParticle*>& p_particles)
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

				Vector3f xixj = xi-xj;
				
				p_particles[i]->m_acceleration += 2.0 * m_viscosity * (p_particles[idx]->m_mass / density_j) 
					* (vi - vj) * (xixj.dot(m_kernel.Cubic_Kernel_Gradient(xi - xj))) / (xixj.squaredNorm() + 0.01*h2);
			}
		}
	}
}

void IISPHWorld::ComputeSurfaceTension()
{
	// skip
}

void IISPHWorld::VelocityAdvection(std::vector<FParticle*>& p_particles, float p_timeStep)
{
	unsigned int numParticles = p_particles.size();
	float h = p_timeStep;

	// Predict v_adv
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			p_particles[i]->m_velocity += h * p_particles[i]->m_acceleration;
		}
	}
}

void IISPHWorld::PredictAdvection(std::vector<FParticle*>& p_particles, std::vector<Vector3f>& p_boundaryParticles, std::vector<float>& p_boundaryPsi, float p_timeStep)
{
	unsigned int numParticles = p_particles.size();
	float h = p_timeStep;

	// Predict v_adv
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			// Vector3r &vel = m_model->getVelocity(0, i);
			// const Vector3r &accel = m_model->getAcceleration(i);
			// vel += h * accel;

			// Compute d_ii
			m_dii[i].setZero();
			float density2 = p_particles[i]->m_density * p_particles[i]->m_density;
			Vector3f &xi = p_particles[i]->m_curPosition;
			for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborList[j];
				Vector3f r = xi - p_particles[idx]->m_curPosition;

				m_dii[i] -= p_particles[idx]->m_mass / density2 * m_kernel.Cubic_Kernel_Gradient(r);
			}
			for (unsigned int j = 0; j < p_particles[i]->m_neighborBoundaryList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborBoundaryList[j];
				Vector3f r = xi - p_boundaryParticles[idx];

				m_dii[i] -= p_boundaryPsi[idx] / density2 * m_kernel.Cubic_Kernel_Gradient(r);

			}
		}
	}

	// Compute rho_adv
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			float density = p_particles[i]->m_density;
			float &density_adv = m_density_adv[i];
			density_adv = density;
			Vector3f& xi = p_particles[i]->m_curPosition;
			Vector3f& vi = p_particles[i]->m_velocity;
			for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborList[j];
				Vector3f& xj = p_particles[idx]->m_curPosition;
				Vector3f& vj = p_particles[idx]->m_velocity;
				
				density_adv += h*p_particles[idx]->m_mass * (vi - vj).dot(m_kernel.Cubic_Kernel_Gradient(xi - xj));
			}
			for (unsigned int j = 0; j < p_particles[i]->m_neighborBoundaryList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborBoundaryList[j];
				Vector3f& xj = p_boundaryParticles[idx];

				density_adv += h*p_boundaryPsi[idx] * (vi).dot(m_kernel.Cubic_Kernel_Gradient(xi - xj));
			}

			float& pressure = m_pressure[i];
			float& lastPressure = m_lastPressure[i];
			lastPressure = 0.5*pressure;

			// Compute a_ii
			float& aii = m_aii[i];
			aii = 0.0;
			Vector3f& dii = m_dii[i];

			float dpi = p_particles[i]->m_mass / (density*density);
			for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborList[j];
				Vector3f& xj = p_particles[idx]->m_curPosition;

				Vector3f kernel = m_kernel.Cubic_Kernel_Gradient(xi - xj);
				Vector3f dji = dpi * kernel;
				aii += p_particles[idx]->m_mass * (dii - dji).dot(kernel);
			}
			for (unsigned int j = 0; j < p_particles[i]->m_neighborBoundaryList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborBoundaryList[j];
				Vector3f& xj = p_boundaryParticles[idx];

				Vector3f kernel = m_kernel.Cubic_Kernel_Gradient(xi - xj);
				Vector3f dji = dpi * kernel;
				aii += p_boundaryPsi[idx] * (dii - dji).dot(kernel);
			}
		}
	}
}
void IISPHWorld::PressureSolve(std::vector<FParticle*>& p_particles, std::vector<Vector3f>& p_boundaryParticles, std::vector<float>& p_boundaryPsi, float p_timeStep)
{
	unsigned int numParticles = p_particles.size();
	float h = p_timeStep;
	float h2 = h*h;

	int m_iterations = 0;
	float omega = 0.5;
	float eta = m_maxError * 0.01 * m_restDensity;  // maxError is given in percent

	float avg_density = 0.0;
	while ((((avg_density - m_restDensity) > eta) || (m_iterations < 2)) && (m_iterations < m_maxIterations))
	{
		avg_density = 0.0;

		// Compute dij_pj
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				Vector3f &dij_pj = m_dij_pj[i];
				dij_pj.setZero();
				Vector3f &xi = p_particles[i]->m_curPosition;
				
				for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
				{
					unsigned int idx = p_particles[i]->m_neighborList[j];
					Vector3f &xj = p_particles[idx]->m_curPosition;
					float densityj = p_particles[idx]->m_density;
					
					dij_pj -= p_particles[idx]->m_mass / (densityj*densityj) * m_lastPressure[idx] * m_kernel.Cubic_Kernel_Gradient(xi - xj);
					
				}
			}
		}

		// Compute new pressure
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				float &aii = m_aii[i];
				float density = p_particles[i]->m_density;
				Vector3f &xi = p_particles[i]->m_curPosition;
				float dpi = p_particles[i]->m_mass / (density*density);
				float sum = 0.0;
				for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
				{
					unsigned int idx = p_particles[i]->m_neighborList[j];
					Vector3f &xj = p_particles[idx]->m_curPosition;

					Vector3f &d_jk_pk = m_dij_pj[idx];

					Vector3f kernel = m_kernel.Cubic_Kernel_Gradient(xi - xj);
					Vector3f dji = dpi * kernel;
					Vector3f d_ji_pi = dji * m_lastPressure[i];

					// \sum ( mj * (\sum dij*pj - djj*pj - \sum_{k \neq i} djk*pk) * m_model->gradW)
					sum += p_particles[idx]->m_mass * (m_dij_pj[i] - m_dii[idx]*m_lastPressure[idx] - (d_jk_pk - d_ji_pi)).dot(kernel);
				}
				for (unsigned int j = 0; j < p_particles[i]->m_neighborBoundaryList.size(); j++)
				{
					unsigned int idx = p_particles[i]->m_neighborBoundaryList[j];
					Vector3f &xj = p_boundaryParticles[idx];
					sum += p_boundaryPsi[idx] * m_dij_pj[i].dot(m_kernel.Cubic_Kernel_Gradient(xi - xj));
				}

				float b = m_restDensity - m_density_adv[i];

				float &pi = m_pressure[i];
				float &lastPi = m_lastPressure[i];
				float denom = aii*h2;
				if (fabs(denom) > 1.0e-9)
					pi = max((1.0 - omega)*lastPi + omega / denom * (b - h2*sum), 0.0);
				else
					pi = 0.0;

				if (pi != 0.0)
				{
					float newDensity = (aii*pi + sum)*h2 - b + m_restDensity;

#pragma omp atomic
					avg_density += newDensity;
				}
				else
				{
#pragma omp atomic
					avg_density += m_restDensity;
				}
			}
		}

		for (int i = 0; i < (int)numParticles; i++)
		{
			float& pi = m_pressure[i];
			float& lastPi = m_lastPressure[i];
			lastPi = pi;
		}

		avg_density /= numParticles;

		m_iterations++;
	}
}
void IISPHWorld::Integration(std::vector<FParticle*>& p_particles, std::vector<Vector3f>& p_boundaryParticles, std::vector<float>& p_boundaryPsi, float p_timeStep)
{
	unsigned int numParticles = p_particles.size();

	// Compute pressure forces
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			Vector3f& xi = p_particles[i]->m_curPosition;
			float& density_i = p_particles[i]->m_density;

			Vector3f& ai = m_pressureAccel[i];
			ai.setZero();

			float dpi = m_pressure[i] / (density_i*density_i);
			for (unsigned int j = 0; j < p_particles[i]->m_neighborList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborList[j];
				Vector3f &xj = p_particles[idx]->m_curPosition;

				// Pressure 
				float& density_j = p_particles[idx]->m_density;
				float dpj = m_pressure[idx] / (density_j*density_j);
				ai -= p_particles[idx]->m_mass * (dpi + dpj) * m_kernel.Cubic_Kernel_Gradient(xi - xj);
			}
			for (unsigned int j = 0; j < p_particles[i]->m_neighborBoundaryList.size(); j++)
			{
				unsigned int idx = p_particles[i]->m_neighborBoundaryList[j];
				Vector3f &xj = p_boundaryParticles[idx];
				ai -= p_boundaryPsi[idx] * (dpi) * m_kernel.Cubic_Kernel_Gradient(xi - xj);

				// re-action force for boundary particle 
				//m_model->getForce(particleId.point_set_id, neighborIndex) += m_model->getMass(i) * a;
			}
		}
	}

	float h = p_timeStep;

	// real integration
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			Vector3f &pos = p_particles[i]->m_curPosition;
			Vector3f &vel = p_particles[i]->m_velocity; 
			vel += m_pressureAccel[i] * h;
			pos += vel * h;
		}
	}
}