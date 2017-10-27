#pragma once
#ifndef __FLUID_KERNEL_H__
#define __FLUID_KERNEL_H__

#include <math.h>
#include <Eigen/Dense>

class FluidKernel
{
public:
	const float PIF = 3.141592f;
	float h;
	float h3;
	float m_k;
	float m_l;
	float res_0;

	float Poly6_Kernel(Vector3f r, float  h)
	{
		float rLength = r.norm();

		if (rLength >= 0 && rLength <= h)
		{
			float a = 315.0f * powf((h*h - rLength*rLength), 3.0f);
			float b = 64.0f * PIF * powf(h, 9.0f);

			return a / b;
		}
		else
		{
			return 0;
		}
	}
	Vector3f Poly6_Kernel_Gradient(Vector3f r, float  h)
	{
		float rLength = r.norm();
		float a = 945.0f * powf((h*h - rLength*rLength), 2.0f);
		float b = 32.0f * PIF * powf(h, 9.0f);

		float scale = a / b;

		return -1.0f * r * scale;
	}
	float Poly6_Kernel_Laplacian(Vector3f r, float  h)
	{
		float rLength = r.norm();
		float a = (h*h - rLength*rLength);
		float b = 8.0f * PIF * powf(h, 9.0f);
		float c = (rLength*rLength - 0.75f * a);

		return (945.0f * a * c) / b;
	}

	float Spiky_Kernel(Vector3f r, float  h)
	{
		float rLength = r.norm();
		if (rLength >= 0 && rLength <= h)
		{
			float a = 15.0f * powf((h - rLength), 3.0f);
			float b = PIF * powf(h, 6.0f);

			return a / b;
		}
		else
		{
			return 0;
		}
	}
	Vector3f Spiky_Kernel_Gradient(Vector3f r, float  h)
	{
		float rLength = r.norm();
		float a = 45.0f * powf((h - rLength), 2.0f);
		float b = PIF * powf(h, 6.0f) * rLength;

		float scale = a / b;

		return -1.0f * r * scale;
	}

	float Viscosity_Kernel(Vector3f r, float  h)
	{
		float rLength = r.norm();

		if (rLength >= 0 && rLength <= h)
		{
			float a = powf(h, 3.0f);
			float b = powf(rLength, 3.0f);
			float c = (-b / (2.0f * a)) + ((rLength*rLength) / (h*h)) + (h / (2.0f * rLength)) - 1.0f;

			return (15.0f * c) / (2.0f * PIF * a);
		}
		else
		{
			return 0;
		}
	}
	Vector3f Viscosity_Kernel_Gradient(Vector3f r, float  h)
	{
		float rLength = r.norm();
		float a = powf(h, 3.0f);
		float b = powf(rLength, 3.0f);
		float c = ((-3.0f * rLength) / (2.0f * a)) + (2.0f / (h*h)) - (h / (2.0f * b));

		float scale = (15.0f * c) / (2.0f * PIF * a);

		return r * scale;
	}
	float Viscosity_Kernel_Laplacian(Vector3f r, float  h)
	{
		float rLength = r.norm();
		float up = 45.0f * (1.0f - (rLength / h));
		float down = PIF * powf(h, 5.0f);

		return up / down;
	}

	void SetSmoothingRadius(float p_radius)
	{
		h = p_radius;
		h3 = h * h * h;
		m_k = 8.0f / (PIF*h3);
		m_l = 48.0f / (PIF*h3);
		res_0 = Cubic_Kernel(Vector3f(0.0f, 0.0f, 0.0f));
	}
	float Cubic_Kernel(Vector3f r)
	{
		float rLength = r.norm();
		float q = rLength / h;
		float res;

		if (q <= 0.5)
		{
			float q2 = q*q;
			float q3 = q2*q;

			res = m_k * (6.0f*q3 - 6.0f*q2 + 1.0f);
		}
		else
		{
			res = m_k * (2.0f*pow(1.0f - q, 3));
		}

		return res;
	}
	float Cubic_Kernel0()
	{
		return res_0;
	}
	Vector3f Cubic_Kernel_Gradient(Vector3f r)
	{
		Vector3f res;
		float rl = r.norm();
		float q = rl / h;

		if (rl > 1.0e-6)
		{
			Vector3f gradq = r * ((float) 1.0 / (rl * h));
			if (q <= 0.5f)
			{
				res = m_l*q*((float) 3.0*q - (float) 2.0)*gradq;
			}
			else
			{
				const float factor = 1.0f - q;
				res = m_l*(-factor*factor)*gradq;
			}
		}

		return res;
	}

	float GetSmoothingRadius() { return h; }
};

#endif