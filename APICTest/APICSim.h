#pragma once
#ifndef __APICSIM_H__
#define __APICSIM_H__

#include <Eigen/Dense>
#include <vector>
#include "Array3d.h"
#include "..\PositionBasedFluid\FluidWorld.h"

using namespace APICArray3d;
using namespace Eigen;
using namespace std;

class APICSim
{
public:
	APICSim() {};
	void Initialize(Vector3f p_origin, Vector3f p_bSize, Vector3i p_nCount, float p_rho);
	void AssignCells(FluidWorld* p_world);
	void GetNeigboringParticles_cell(int i, int j, int k, int wl, int wh, int hl, int hh, int dl, int dh, std::vector<FParticle *>& res);
	void Map_P2G(FluidWorld* p_world);
	Vector3f GetNiNjNk() { return Vector3f(ni, nj, nk); }
	Vector3f GetDxDyDz() { return Vector3f(dx, dy, dz); }
	Vector3f& GetGridPos(int i, int j, int k) { return cells_pos[(k*nj*ni) + j*ni + i]; }
	Vector3f GetVelocity(Vector3f& pos);
	void UpdateAffineMatrix(FluidWorld* p_world);

private:

	float rho;
	Vector3f origin;
	int ni, nj, nk;
	float dx, dy, dz;

	APICArray3d::Array3d<float> u;
	APICArray3d::Array3d<float> v;
	APICArray3d::Array3d<float> w;
	
	std::vector<std::vector<FParticle*>> cells;
	std::vector<Vector3f> cells_pos;
	

	inline float interpolate_value(Vector3f& point, APICArray3d::Array3d<float>& grid);
	inline Vector3f affine_interpolate_value(Vector3f& point, APICArray3d::Array3d<float>& grid);
};

inline float linear_kernel(const Vector3f& d, const float& h)
{
	return std::max((1.0 - fabs(d(0) / h)) * (1.0 - fabs(d(1) / h)) * (1.0 - fabs(d(2) / h)), 0.0);
}

template<class T>
inline void get_barycentric(T x, int& i, T& f, int i_low, int i_high)
{
	T s = std::floor(x);
	i = (int)s;
	if (i<i_low) {
		i = i_low;
		f = 0;
	}
	else if (i>i_high - 2) {
		i = i_high - 2;
		f = 1;
	}
	else
		f = (T)(x - s);
}

template<class S, class T>
inline S lerp(const S& value0, const S& value1, T f)
{
	return (1 - f)*value0 + f*value1;
}

template<class S, class T>
inline S bilerp(const S& v00, const S& v10,
	const S& v01, const S& v11,
	T fx, T fy)
{
	return lerp(lerp(v00, v10, fx),
		lerp(v01, v11, fx),
		fy);
}

template<class S, class T>
inline S trilerp(const S& v00, const S& v01, const S& v02, const S& v03,
	const S& v10, const S& v11, const S& v12, const S& v13,
	T fx, T fy, T fz)
{
	return lerp(
		lerp(lerp(v00, v01, fx), lerp(v02, v03, fx), fy),
		lerp(lerp(v10, v11, fx), lerp(v12, v13, fx), fy),
		fz);
}

template<class T>
inline Eigen::Matrix<T, 3, 1> grad_trilerp(const T& v000, const T& v100, const T& v010, const T& v110,
	const T& v001, const T& v101, const T& v011, const T& v111,
	T fx, T fy, T fz)
{

	return  Eigen::Matrix<T, 3, 1>(fy - 1.0, fz - 1.0, fx - 1.0) * v000 +
			Eigen::Matrix<T, 3, 1>(1.0 - fy, fz - 1.0, -fx) * v100 +
			Eigen::Matrix<T, 3, 1>(-fy, 1.0 - fx, fx - 1.0) * v010 +
			Eigen::Matrix<T, 3, 1>(fy, 1.0 - fz, -fx) * v110 +
			Eigen::Matrix<T, 3, 1>(fy - 1.0, -fz, 1.0 - fx) * v001 +
			Eigen::Matrix<T, 3, 1>(1.0 - fy, -fz, fx) * v101 +
			Eigen::Matrix<T, 3, 1>(-fy, fz, 1.0 - fx) * v011 +
			Eigen::Matrix<T, 3, 1>(fy, fz, fx) * v111;
}


#endif