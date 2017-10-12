#include "FluidWorld.h"

FluidWorld::FluidWorld()
{
	m_accTimeIntegration = 0.0f;
	m_timeStep = 0.0025f;

	m_viscosity = 0.002f;
	m_surfaceTensionThr = 0.2f;
	m_surfaceTensionCoeff = 0.2f;
	m_useGravity = false;
	m_restDensity = 1000.0f;
}
FluidWorld::~FluidWorld() {}

void FluidWorld::CreateParticles(std::vector<Vector3f>& p_damParticles, std::vector<Vector3f>& p_containerParticles, float p_particleRadius)
{
	m_numOfParticles = (unsigned int)p_damParticles.size();
	m_numOfBoundaryParticles = (unsigned int)p_containerParticles.size();
	m_particleRadius = p_particleRadius;
	m_smoothingLength = 4.0f * m_particleRadius;
	k.SetSmoothingRadius(m_smoothingLength);

	float diameter = 2.0f * p_particleRadius;

	m_particles.resize(m_numOfParticles);
	m_particlesLambda.resize(m_numOfParticles);
	m_boundaryParticles.resize(p_containerParticles.size());
	m_boundaryPsi.resize(p_containerParticles.size());

#pragma omp parallel default(shared)
	{
		// dam particles creation
#pragma omp for schedule(static)
		for (unsigned int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i] = new FParticle(i);
			m_particles[i]->m_mass = 0.8f * m_restDensity * diameter * diameter * diameter;
			m_particles[i]->m_restPosition = p_damParticles[i];
			m_particles[i]->m_curPosition = p_damParticles[i];

			m_particles[i]->m_velocity.setZero();
			m_particles[i]->m_acceleration.setZero();
		}

		// copy boundary particles
#pragma omp for schedule(static)
		for (unsigned int i = 0; i < m_numOfBoundaryParticles; i++)
		{
			m_boundaryParticles[i] = p_containerParticles[i];
		}

		// boudary particles Psi value 
#pragma omp for schedule(static)
		for (unsigned int i = 0; i < m_numOfBoundaryParticles; i++)
		{
			float delta = k.Cubic_Kernel0();
			for (unsigned int j = 0; j < m_numOfBoundaryParticles; j++)
			{
				Vector3f r = m_boundaryParticles[i] - m_boundaryParticles[j];
				if (i != j && r.norm() <= m_smoothingLength)
				{
					delta += k.Cubic_Kernel(r);
				}
			}
			float volume = 1.0f / delta;
			m_boundaryPsi[i] = m_restDensity * volume;
		}
	}

	if(p_particleRadius == 0.025f)
		debugFlag = true;
}

void FluidWorld::Reset()
{
	for (unsigned int i = 0; i < m_numOfParticles; i++)
	{
		m_particles[i]->m_acceleration.setZero();
		m_particles[i]->m_velocity.setZero();
		m_particles[i]->m_curPosition = m_particles[i]->m_restPosition;
	}
}

void FluidWorld::AddFParticle(Vector3f p_position, Vector3f p_velocity)
{
	float diameter = 2.0f * m_particleRadius;
	int pid = m_numOfParticles;

	FParticle * aP = new FParticle(pid);
	aP->m_mass = 0.8f * m_restDensity * diameter * diameter * diameter;
	aP->m_restPosition = aP->m_curPosition = p_position;
	aP->m_velocity = p_velocity;
	aP->m_acceleration.setZero();

	m_particles.push_back(aP);
	m_numOfParticles += 1;
	m_particlesLambda.resize(m_numOfParticles);
}

void FluidWorld::DeleteFParticle(int p_id)
{
	int idx = -1;
	for (int i = 0; i < m_numOfParticles; i++)
	{
		if (m_particles[i]->m_pid == p_id)
		{
			idx = i;
			break;
		}
	}

	if (idx != -1)
	{
		m_particles.erase(m_particles.begin() + idx);

		m_numOfParticles = m_particles.size();
		for (int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i]->m_pid = i;
		}
		m_particlesLambda.resize(m_numOfParticles);
	}
}
void FluidWorld::DeleteAll()
{
	m_numOfParticles = 0;
	m_particles.clear();
	m_particles.resize(0);
	m_particlesLambda.clear();
	m_particlesLambda.resize(0);
}

void FluidWorld::NeighborListUpdate()
{
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)
		for (unsigned int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i]->m_neighborList.clear();
			m_particles[i]->m_neighborBoundaryList.clear();
			m_particles[i]->m_neighborList.resize(0);
			m_particles[i]->m_neighborBoundaryList.resize(0);

			for (unsigned int j = 0; j < m_numOfParticles; j++)
			{
				Vector3f r = m_particles[i]->m_curPosition - m_particles[j]->m_curPosition;
				if (i != j && r.norm() <= m_smoothingLength)
					m_particles[i]->m_neighborList.push_back(j);
			}
			for (unsigned int j = 0; j < m_numOfBoundaryParticles; j++)
			{
				Vector3f r = m_particles[i]->m_curPosition - m_boundaryParticles[j];
				if (r.norm() <= m_smoothingLength)
					m_particles[i]->m_neighborBoundaryList.push_back(j);
			}
		}
	}
}

void FluidWorld::StepPBF()
{
	float h = m_timeStep;

	// clear ExternForce
	for (unsigned int i = 0; i < m_numOfParticles; i++)
	{
		m_particles[i]->m_acceleration = Vector3f(0.0f, -9.8f, 0.0f);

		m_particles[i]->m_deltaX.setZero();
		m_particles[i]->m_oldPosition = m_particles[i]->m_curPosition;

		if (m_particles[i]->m_mass != 0.0f)
		{
			m_particles[i]->m_velocity += m_particles[i]->m_acceleration * h;
			m_particles[i]->m_curPosition += m_particles[i]->m_velocity * h;
		}
	}

	NeighborListUpdate();

	ConstraintProjection();

	
	for (unsigned int i = 0; i < m_numOfParticles; i++)
	{
		m_particles[i]->m_velocity = (m_particles[i]->m_curPosition - m_particles[i]->m_oldPosition) * (1.0f / h);
		
	}

	ComputeXSPHViscosity();

	UpdateTimeStepSizeCFL();

	m_accTimeIntegration += h;
}

void FluidWorld::ConstraintProjection()
{
	int maxiter = 100;
	int iter = 0;

	float eps = 1.0e-6;

	unsigned int numParticles = m_numOfParticles;
	float invH = 1.0 / m_timeStep;
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
			for (unsigned int i = 0; i < m_numOfParticles; i++)
			{
				// computePBFDensity
				m_particles[i]->m_curDensity = m_particles[i]->m_mass * k.Cubic_Kernel0();

				for (unsigned int j = 0; j < m_particles[i]->m_neighborList.size(); j++)
				{
					unsigned int idx = m_particles[i]->m_neighborList[j];
					Vector3f r = m_particles[i]->m_curPosition - m_particles[idx]->m_curPosition;

					m_particles[i]->m_curDensity += m_particles[idx]->m_mass * k.Cubic_Kernel(r);
				}

				for (unsigned int j = 0; j < m_particles[i]->m_neighborBoundaryList.size(); j++)
				{
					unsigned int idx = m_particles[i]->m_neighborBoundaryList[j];
					Vector3f r = m_particles[i]->m_curPosition - m_boundaryParticles[idx];
					m_particles[i]->m_curDensity += m_boundaryPsi[idx] * k.Cubic_Kernel(r);
				}

				float density_err = std::max(m_particles[i]->m_curDensity, density0) - density0;
#pragma omp atomic
				avg_density_err += density_err / m_numOfParticles;

				// Evaluate constraint function
				float C = std::max(m_particles[i]->m_curDensity / density0 - 1.0f, 0.0f);

				if (C != 0.0f)
				{
					// Compute gradients dC/dx_j 
					float sum_grad_C2 = 0.0;
					Vector3f gradC_i(0.0f, 0.0f, 0.0f);

					for (unsigned int j = 0; j < m_particles[i]->m_neighborList.size(); j++)
					{
						unsigned int idx = m_particles[i]->m_neighborList[j];
						Vector3f r = m_particles[i]->m_curPosition - m_particles[idx]->m_curPosition;

						Vector3f gradC_j = -m_particles[idx]->m_mass / m_restDensity * k.Cubic_Kernel_Gradient(r);
						sum_grad_C2 += gradC_j.squaredNorm();
						gradC_i -= gradC_j;
					}

					for (unsigned int j = 0; j < m_particles[i]->m_neighborBoundaryList.size(); j++)
					{
						unsigned int idx = m_particles[i]->m_neighborBoundaryList[j];
						Vector3f r = m_particles[i]->m_curPosition - m_boundaryParticles[idx];

						Vector3f gradC_j = -m_boundaryPsi[idx] / m_restDensity * k.Cubic_Kernel_Gradient(r);
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
			for (unsigned int i = 0; i < m_numOfParticles; i++)
			{
				Vector3f corr(0.0f, 0.0f, 0.0f);

				for (unsigned int j = 0; j < m_particles[i]->m_neighborList.size(); j++)
				{
					unsigned int idx = m_particles[i]->m_neighborList[j];
					Vector3f r = m_particles[i]->m_curPosition - m_particles[idx]->m_curPosition;

					Vector3f gradC_j = -m_particles[idx]->m_mass / m_restDensity * k.Cubic_Kernel_Gradient(r);
					corr -= (m_particlesLambda[i] + m_particlesLambda[idx]) * gradC_j;
				}

				for (unsigned int j = 0; j < m_particles[i]->m_neighborBoundaryList.size(); j++)
				{
					unsigned int idx = m_particles[i]->m_neighborBoundaryList[j];
					Vector3f r = m_particles[i]->m_curPosition - m_boundaryParticles[idx];

					Vector3f gradC_j = -m_boundaryPsi[idx] / m_restDensity * k.Cubic_Kernel_Gradient(r);
					corr -= (m_particlesLambda[i]) * gradC_j;
				}

				m_particles[i]->m_deltaX = corr;
			}

#pragma omp for schedule(static)
			for (unsigned int i = 0; i < m_numOfParticles; i++)
			{
				m_particles[i]->m_curPosition += m_particles[i]->m_deltaX;
			}
		}

		iter++;
	}
}

void FluidWorld::ComputeXSPHViscosity()
{
	// Compute viscosity forces (XSPH)
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (unsigned int i = 0; i < m_numOfParticles; i++)
		{
			for (unsigned int j = 0; j < m_particles[i]->m_neighborList.size(); j++)
			{
				unsigned int idx = m_particles[i]->m_neighborList[j];
				Vector3f r = m_particles[i]->m_curPosition - m_particles[idx]->m_curPosition;

				Vector3f velCorr = m_viscosity * (m_particles[idx]->m_mass / m_particles[idx]->m_curDensity)
					* (m_particles[i]->m_velocity - m_particles[idx]->m_velocity) * k.Cubic_Kernel(r);

				m_particles[i]->m_velocity -= velCorr;
			}
		}
	}
}

void FluidWorld::UpdateTimeStepSizeCFL()
{
	float radius = m_particleRadius;
	float h = m_timeStep;

	// Approximate max. position change due to current velocities
	float maxVel = 0.1;
	unsigned int numParticles = m_numOfParticles;
	float diameter = 2.0*radius;
	for (unsigned int i = 0; i < numParticles; i++)
	{
		Vector3f vel = GetParticle(i)->m_velocity;
		Vector3f accel = GetParticle(i)->m_acceleration;
		float velMag = (vel + accel*h).squaredNorm();
		if (velMag > maxVel)
			maxVel = velMag;
	}

	// boundary particles
	/*
	for (unsigned int i = 0; i < m_model->numberOfRigidBodyParticleObjects(); i++)
	{
	FluidModel::RigidBodyParticleObject *rbpo = m_model->getRigidBodyParticleObject(i);
	if (rbpo->m_rigidBody->isDynamic())
	{
	for (unsigned int j = 0; j < rbpo->numberOfParticles(); j++)
	{
	const Vector3r &vel = rbpo->m_v[j];
	const Real velMag = vel.squaredNorm();
	if (velMag > maxVel)
	maxVel = velMag;
	}
	}
	}
	*/

	// Approximate max. time step size 		
	float m_cflFactor = 0.5;
	float m_cflMaxTimeStepSize = 0.005;
	float minTimeStepSize = 0.0001;
	h = m_cflFactor * .4 * (diameter / (sqrt(maxVel)));

	h = std::min(h, m_cflMaxTimeStepSize);
	h = std::max(h, minTimeStepSize);

	m_timeStep = h;
}