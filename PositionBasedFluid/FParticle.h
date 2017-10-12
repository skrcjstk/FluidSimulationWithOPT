#pragma once
#ifndef __FLUID_PARTICLE_H__
#define __FLUID_PARTICLE_H__

#include <Eigen/Dense>
#include <vector>
#include <utility>

using namespace Eigen;

class FParticle
{
public:
	unsigned int m_pid;
	float	m_mass;

	Vector3f m_restPosition;
	Vector3f m_oldPosition;
	Vector3f m_curPosition;
	Vector3f m_goalPosition;
	Vector3f m_deltaX;

	Vector3f m_velocity;
	Vector3f m_intermediateVelocity;
	Vector3f m_acceleration;

	float m_goalCount;
	float m_curDensity;
	float m_kP = 0.2f;

	float m_pressure;

	std::vector<unsigned int> m_neighborList;
	std::vector<unsigned int> m_neighborBoundaryList;

	std::vector<unsigned int> m_connectedParticleIdx;
	std::vector<float> m_distanceToConnectedParticle;


	FParticle(unsigned int p_pid)
	{
		m_pid = p_pid;
	}
};

#endif
