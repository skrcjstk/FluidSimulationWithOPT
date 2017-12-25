#include "FluidWorld.h"

FluidWorld::FluidWorld()
{
	m_accTimeIntegration = 0.0f;
	m_timeStep = 0.005f;

	m_useGravity = false;
	m_restDensity = 1000.0f;
	
#if FLUID_METHOD == 0
	pbfWorld = new PBFWorld(m_restDensity, 0.02f, 0.2f, 0.2f);
#elif FLUID_METHOD == 1
	iisphWorld = new IISPHWorld(m_restDensity, 0.02f, 0.2f, 0.2f, 0.01f, 100);
#elif FLUID_METHOD == 2
	wcsphWorld = new WCSPHWorld(m_restDensity, 0.02f, 0.2f, 0.2f, 0.01f, 100);
#endif

}
FluidWorld::~FluidWorld() {}

void FluidWorld::CreateParticles(std::vector<Vector3f>& p_damParticles, std::vector<Vector3f>& p_containerParticles, float p_particleRadius)
{
	m_numOfParticles = p_damParticles.size();
	m_numOfBoundaryParticles = p_containerParticles.size();
	m_particleRadius = p_particleRadius;
	m_smoothingLength = 4.0f * m_particleRadius;
	k.SetSmoothingRadius(m_smoothingLength);

#if FLUID_METHOD == 0
	pbfWorld->SetSmoothingLength(m_smoothingLength);
	pbfWorld->InitializeSimulationData(m_numOfParticles);
#elif FLUID_METHOD == 1
	iisphWorld->SetSmoothingLength(m_smoothingLength);
	iisphWorld->InitializeSimulationData(m_numOfParticles);
#elif FLUID_METHOD == 2
	wcsphWorld->SetSmoothingLength(m_smoothingLength);
	wcsphWorld->InitializeSimulationData(m_numOfParticles);
#endif

	float diameter = 2.0f * p_particleRadius;

	m_particles.resize(m_numOfParticles);
	m_boundaryParticles.resize(p_containerParticles.size());
	
#pragma omp parallel default(shared)
	{
		// dam particles creation
#pragma omp for schedule(static)
		for (int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i] = new FParticle();
			m_particles[i]->m_mass = 0.8f * m_restDensity * diameter * diameter * diameter;
			m_particles[i]->m_restPosition = p_damParticles[i];
			m_particles[i]->m_curPosition = p_damParticles[i];

			m_particles[i]->m_velocity.setZero();
			m_particles[i]->m_acceleration.setZero();
			m_particles[i]->m_c.setZero();
		}

		// copy boundary particles
#pragma omp for schedule(static)
		for (int i = 0; i < m_numOfBoundaryParticles; i++)
		{
			m_boundaryParticles[i] = new FParticle();
			m_boundaryParticles[i]->m_mass = 0.8f * m_restDensity * diameter * diameter * diameter;
			m_boundaryParticles[i]->m_restPosition = p_containerParticles[i];
			m_boundaryParticles[i]->m_curPosition = p_containerParticles[i];

			m_boundaryParticles[i]->m_velocity.setZero();
			m_boundaryParticles[i]->m_acceleration.setZero();
			m_boundaryParticles[i]->m_c.setZero();
		}

		// boudary particles Psi value 
#pragma omp for schedule(static)
		for (int i = 0; i < m_numOfBoundaryParticles; i++)
		{
			float delta = k.Cubic_Kernel0();
			for (int j = 0; j < m_numOfBoundaryParticles; j++)
			{
				Vector3f r = m_boundaryParticles[i]->m_restPosition - m_boundaryParticles[j]->m_restPosition;
				if (i != j && r.norm() <= m_smoothingLength)
				{
					delta += k.Cubic_Kernel(r);
				}
			}
			float volume = 1.0f / delta;
			m_boundaryParticles[i]->m_mass = m_restDensity * volume;
		}
	}

	if(p_particleRadius == 0.025f)
		debugFlag = true;
}
void FluidWorld::Reset()
{
	for (int i = 0; i < m_numOfParticles; i++)
	{
		m_particles[i]->m_acceleration.setZero();
		m_particles[i]->m_velocity.setZero();
		m_particles[i]->m_curPosition = m_particles[i]->m_restPosition;
	}
#if FLUID_METHOD == 0
	pbfWorld->Reset();
#elif FLUID_METHOD == 1
	iisphWorld->Reset();
#elif FLUID_METHOD == 2
	wcsphWorld->Reset();
#endif

}
void FluidWorld::AddFParticle(Vector3f p_position, Vector3f p_velocity)
{
	float diameter = 2.0f * m_particleRadius;

	FParticle * aP = new FParticle();
	aP->m_mass = 0.8f * m_restDensity * diameter * diameter * diameter;
	aP->m_restPosition = aP->m_curPosition = p_position;
	aP->m_velocity = p_velocity;
	aP->m_acceleration.setZero();

	m_particles.push_back(aP);
	m_numOfParticles += 1;

#if FLUID_METHOD == 0
	pbfWorld->InitializeSimulationData(m_numOfParticles);
#elif FLUID_METHOD == 1
	iisphWorld->InitializeSimulationData(m_numOfParticles);
#elif FLUID_METHOD == 2
	wcsphWorld->InitializeSimulationData(m_numOfParticles);
#endif
}
void FluidWorld::DeleteFParticle(int p_idx)
{
	m_particles.erase(m_particles.begin() + p_idx);
	m_numOfParticles = m_particles.size();

#if FLUID_METHOD == 0
	pbfWorld->InitializeSimulationData(m_numOfParticles);
#elif FLUID_METHOD == 1
	iisphWorld->InitializeSimulationData(m_numOfParticles);
#elif FLUID_METHOD == 2
	wcsphWorld->InitializeSimulationData(m_numOfParticles);
#endif
}
void FluidWorld::DeleteAll()
{
	m_numOfParticles = 0;
	m_particles.clear();
	m_particles.resize(0);

#if FLUID_METHOD == 0
	pbfWorld->InitializeSimulationData(0);
#elif FLUID_METHOD == 1
	iisphWorld->InitializeSimulationData(0);
#elif FLUID_METHOD == 2
	wcsphWorld->InitializeSimulationData(0);
#endif
}
void FluidWorld::NeighborListUpdate()
{
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)
		for (int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i]->m_neighborList.clear();
			m_particles[i]->m_neighborBoundaryList.clear();
			m_particles[i]->m_neighborList.resize(0);
			m_particles[i]->m_neighborBoundaryList.resize(0);

			for (int j = 0; j < m_numOfParticles; j++)
			{
				Vector3f r = m_particles[i]->m_curPosition - m_particles[j]->m_curPosition;
				if (i != j && r.norm() <= m_smoothingLength)
					m_particles[i]->m_neighborList.push_back(j);
			}
			for (int j = 0; j < m_numOfBoundaryParticles; j++)
			{
				Vector3f r = m_particles[i]->m_curPosition - m_boundaryParticles[j]->m_curPosition;
				if (r.norm() <= m_smoothingLength)
					m_particles[i]->m_neighborBoundaryList.push_back(j);
			}
		}
	}
}
void FluidWorld::ComputeDensities()
{
	int numParticles = m_particles.size();

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			// Compute current density for particle i
			m_particles[i]->m_density = m_particles[i]->m_mass * k.Cubic_Kernel0();

			for (int j = 0; j < m_particles[i]->m_neighborList.size(); j++)
			{
				int idx = m_particles[i]->m_neighborList[j];
				Vector3f r = m_particles[i]->m_curPosition - m_particles[idx]->m_curPosition;

				m_particles[i]->m_density += m_particles[idx]->m_mass * k.Cubic_Kernel(r);
			}

			for (int j = 0; j < m_particles[i]->m_neighborBoundaryList.size(); j++)
			{
				int idx = m_particles[i]->m_neighborBoundaryList[j];
				Vector3f r = m_particles[i]->m_curPosition - m_boundaryParticles[idx]->m_curPosition;
				m_particles[i]->m_density += m_boundaryParticles[idx]->m_mass * k.Cubic_Kernel(r);
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
	for (int i = 0; i < numParticles; i++)
	{
		Vector3f vel = GetParticle(i)->m_velocity;
		Vector3f accel = GetParticle(i)->m_acceleration;
		float velMag = (vel + accel*h).squaredNorm();
		if (velMag > maxVel)
			maxVel = velMag;
	}

	// boundary particles (if boundary moving)
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
int  FluidWorld::GetFluidMethodNumber()
{
#if FLUID_METHOD == 0
	return 0;
#elif FLUID_METHOD == 1
	return 1;
#elif FLUID_METHOD == 2
	return 2;
#endif
}
void* FluidWorld::GetFluidMethod()
{
#if FLUID_METHOD == 0
	return pbfWorld;
#elif FLUID_METHOD == 1
	return iisphWorld;
#elif FLUID_METHOD == 2
	return wcsphWorld;
#endif
}

void FluidWorld::StepPBF()
{
	float h = m_timeStep;

	// clear ExternForce
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i]->m_acceleration = Vector3f(0.0f, -9.8f, 0.0f);
			m_particles[i]->m_oldPosition = m_particles[i]->m_curPosition;

			if (m_particles[i]->m_mass != 0.0f)
			{
				m_particles[i]->m_velocity += m_particles[i]->m_acceleration * h;
				m_particles[i]->m_curPosition += m_particles[i]->m_velocity * h;
			}
		}
	}

	NeighborListUpdate();
	pbfWorld->ConstraintProjection(m_particles, m_boundaryParticles, h);
		
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i]->m_velocity = (m_particles[i]->m_curPosition - m_particles[i]->m_oldPosition) * (1.0f / h);
		}
	}

	pbfWorld->ComputeXSPHViscosity(m_particles);

	//UpdateTimeStepSizeCFL();

	m_accTimeIntegration += h;
}
void FluidWorld::StepPBFonFine()
{
	float h = m_timeStep;
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < m_numOfParticles; i++)
		{
			//m_particles[i]->m_deltaX.setZero();
			m_particles[i]->m_oldPosition = m_particles[i]->m_curPosition;

			if (m_particles[i]->m_mass != 0.0f)
			{
				m_particles[i]->m_curPosition += m_particles[i]->m_velocity * h;
			}
		}
	}

	NeighborListUpdate();
	pbfWorld->ConstraintProjection(m_particles, m_boundaryParticles, h);

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i]->m_velocity = (m_particles[i]->m_curPosition - m_particles[i]->m_oldPosition) * (1.0f / h);
		}
	}

	pbfWorld->ComputeXSPHViscosity(m_particles);

	//UpdateTimeStepSizeCFL();

	m_accTimeIntegration += h;
}
void FluidWorld::StepPBFonSub1()
{
	float h = m_timeStep;

	// clear ExternForce
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i]->m_acceleration = Vector3f(0.0f, -9.8f, 0.0f);
			m_particles[i]->m_oldPosition = m_particles[i]->m_curPosition;
			if (m_particles[i]->m_mass != 0.0f)
			{
				m_particles[i]->m_velocity += m_particles[i]->m_acceleration * h;
				m_particles[i]->m_curPosition += m_particles[i]->m_velocity * h;
				m_particles[i]->m_tempPosition = m_particles[i]->m_curPosition;
			}
		}
	}

	NeighborListUpdate();
}
void FluidWorld::StepPBFonSub2()
{
	float h = m_timeStep;

	pbfWorld->ConstraintProjection(m_particles, m_boundaryParticles, h);

#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i]->m_velocity = (m_particles[i]->m_curPosition - m_particles[i]->m_oldPosition) * (1.0f / h);
		}
	}

	pbfWorld->ComputeXSPHViscosity(m_particles);

	//UpdateTimeStepSizeCFL();

	m_accTimeIntegration += h;
}

void FluidWorld::StepPBFonSub1WithTF()
{
	float h = m_timeStep;

#pragma omp parallel default(shared)
	{
		// clear ExternForce
#pragma omp for schedule(static)  
		for (int i = 0; i < m_numOfParticles; i++)
		{
			m_particles[i]->m_acceleration = Vector3f(0.0f, -9.8f, 0.0f);
			m_particles[i]->m_oldPosition = m_particles[i]->m_curPosition;
			if (m_particles[i]->m_mass != 0.0f)
			{
				m_particles[i]->m_velocity += m_particles[i]->m_acceleration * h;
				m_particles[i]->m_curPosition += m_particles[i]->m_velocity * h;
			}
		}
	}
	NeighborListUpdate();
}

void FluidWorld::StepPBFonSub2WithTF()
{
	float h = m_timeStep;
#pragma omp parallel default(shared)
	{
#pragma omp for schedule(static)  
		for (int i = 0; i < m_numOfParticles; i++)
			m_particles[i]->m_velocity = (m_particles[i]->m_curPosition - m_particles[i]->m_oldPosition) * (1.0f / h);
	}
	//UpdateTimeStepSizeCFL();
}
