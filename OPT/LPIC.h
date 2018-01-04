#pragma once
#pragma once
#ifndef __LOCAL_PIC_H__
#define __LOCAL_PIC_H__

#include <Eigen/Dense>
#include <vector>
#include "Array3d.h"
#include "FluidWorld.h"

using namespace APICArray3d;
using namespace Eigen;
using namespace std;

enum Gid { B = -1, A = 0, F = 1 };

class LPIC
{
public:
	LPIC() {};

	void clean();

	void Initialize(Vector3f p_bSize, Vector3i p_nCount, float p_rho);
	void LAPICDesc(float result[], FParticle* p_center, std::vector<FParticle*>& p_list, float p_radii);
	Vector3f GetVelocity(Vector3f& pos);
	float GetMass(Vector3f& pos);

	Vector3i GetNiNjNk() { return Vector3i(ni, nj, nk); }
	Vector3f GetDxDyDz() { return Vector3f(dx, dy, dz); }
	Vector3f& GetGridPos(int i, int j, int k) { return cells_centor_pos[(k*nj*ni) + j*ni + i]; }
	Vector3i& GetAssignResultF(int p_idx) { return AssignResultF[p_idx]; }

private:
	float rho;
	Vector3f origin, center;
	int ni, nj, nk;
	int centerIdx;

	Vector3f toOrigin;

	float dx, dy, dz;

	std::vector<Gid> geo;
	APICArray3d::Array3d<float> u;
	APICArray3d::Array3d<float> v;
	APICArray3d::Array3d<float> w;

	std::vector<std::vector<FParticle*>> cells;
	std::vector<Vector3f> cells_centor_pos;
	std::vector<Vector3f> cells_centor_uvw_coord;
	std::vector<Vector3i> AssignResultF;

	void GetNeigboringParticles_cell(int i, int j, int k, int wl, int wh, int hl, int hh, int dl, int dh, std::vector<FParticle *>& res);
	inline float interpolate_value(Vector3f& point, APICArray3d::Array3d<float>& grid);
	
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
inline S trilerp(const S& v000, const S& v100,
	const S& v010, const S& v110,
	const S& v001, const S& v101,
	const S& v011, const S& v111,
	T fx, T fy, T fz)
{
	return lerp(bilerp(v000, v100, v010, v110, fx, fy),
		bilerp(v001, v101, v011, v111, fx, fy),
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