#pragma once
#ifndef __FLUID_PARTICLE_H__
#define __FLUID_PARTICLE_H__

#include <Eigen/Dense>
#include <vector>
#include <utility>

using namespace Eigen;

enum Pid { Fluid, Boundary };

class FParticle
{
public:
	Pid   m_pid;
	int   m_pIdx;

	float m_mass;
	float m_density;
	
	Vector3f m_restPosition;
	Vector3f m_oldPosition;
	Vector3f m_curPosition;
	Vector3f m_velocity;
	Vector3f m_acceleration;
		
	std::vector<FParticle *> m_neighborList;
	//std::vector<FParticle *> m_neighborBoundaryList;

	Vector3f m_tempPosition;
	Vector3f m_tempVelocity;

	bool m_interpolated = false;

	Matrix3f m_c;
};

#endif
