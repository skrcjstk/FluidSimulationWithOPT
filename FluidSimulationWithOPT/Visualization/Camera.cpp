#include "camera.h"

Camera::Camera()
{
	floorConstraint = FALSE;
	floorBlend = 0.5;
}

Camera::~Camera()
{

}

void Camera::Init(ViewType tp, float dis)		// Camera 초기화
{
	tarX = 0.0f;
	tarY = 0.0f;
	tarZ = 0.0f;
	fov = 45.0f;
	distance = dis;
	
	switch(tp)
	{
		case VIEW_TOP :		// top view의 경우 
			posX = 0.0f;
			posY = 0.0f;
			posZ = distance;
			angOrb = 270.0f;
			angLat = 0.0f;
			angLev = 0.0f;
			norX = 0.0f;
			norY = 1.0f;
			norZ = 0.0f;
			break;

		case VIEW_BOTTOM :
			posX = 0.0f;
			posY = 0.0f;
			posZ = -distance;
			angOrb = 270.0f;
			angLat = -90.0f;
			angLev = 0.0f;
			norX = 0.0f;
			norY = 1.0f;
			norZ = 0.0f;
			break;

		case VIEW_FRONT :
			posX = 0.0f;
			posY = -distance;
			posZ = 0.0f;
			angOrb = 270.0f;
			angLat = 0.0f;
			angLev = 0.0f;
			norX = 0.0f;
			norY = 0.0f;
			norZ = 1.0f;
			break;

		case VIEW_BACK :
			posX = 0.0f;
			posY = distance;
			posZ = 0.0f;
			angOrb = 270.0f;
			angLat = 0.0f;
			angLev = 0.0f;
			norX = 0.0f;
			norY = 0.0f;
			norZ = 1.0f;
			break;

		case VIEW_LEFT :
			posX = -distance;
			posY = 0.0f;
			posZ = 0.0f;
			angOrb = 180.0f;
			angLat = 0.0f;
			angLev = 0.0f;
			norX = 0.0f;
			norY = 1.0f;
			norZ = 0.0f;
			break;

		case VIEW_RIGHT :
			posX = distance;
			posY = 0.0f;
			posZ = 0.0f;
			angOrb = 0.0f;
			angLat = 0.0f;
			angLev = 0.0f;
			norX = 0.0f;
			norY = 1.0f;
			norZ = 0.0f;
			break;

		case VIEW_PERS :
		case VIEW_ORTHO:
			angOrb = 270.0f;
			angLat = 45.0f;
			angLev = 0.0f;
			posX = distance * (float)cos(RADIAN(angLat)) * (float)cos(RADIAN(angOrb));
			posY = distance * (float)sin(RADIAN(angLat));
			posZ = -distance * (float)cos(RADIAN(angLat)) * (float)sin(RADIAN(angOrb));
			norX = 0.0f;
			norY = 1.0f;
			norZ = -1.0f;
			break;
	}
	type = tp;
}

void Camera::EditDistance(float target)				// Camera 의 Distance를 조정
{
	distance += target;
	if(distance < 0.1f)
		distance = 0.1f;
	posX = distance * (float)cos(RADIAN(angLat)) * (float)cos(RADIAN(angOrb)) + tarX;
	posY = distance * (float)sin(RADIAN(angLat)) + tarY;
	posZ = -(distance * (float)cos(RADIAN(angLat)) * (float)sin(RADIAN(angOrb)) - tarZ);
}

void Camera::EditPan(float X, float Y)	// Camera 의 Panning에 따른 조정
{
	// 먼저 카메라의 view vector와 right vector를 알아낸다.
	float viewVec[3], rightVec[3], upVec[3];
	float norm;

	viewVec[0] = tarX - posX;
	viewVec[1] = tarY - posY;
	viewVec[2] = tarZ - posZ;
	norm = 1/sqrt(viewVec[0] * viewVec[0] + viewVec[1] * viewVec[1] + viewVec[2] * viewVec[2]);
	viewVec[0] *= norm;
	viewVec[1] *= norm;
	viewVec[2] *= norm;
	
	rightVec[0] = viewVec[1]*norZ - viewVec[2]*norY;
	rightVec[1] = viewVec[2]*norX - viewVec[0]*norZ;
	rightVec[2] = viewVec[0]*norY - viewVec[1]*norX;
	norm = 1/sqrt(rightVec[0] * rightVec[0] + rightVec[1] * rightVec[1] + rightVec[2] * rightVec[2]);
	rightVec[0] *= norm;
	rightVec[1] *= norm;
	rightVec[2] *= norm;

	upVec[0] = rightVec[1]*viewVec[2] - rightVec[2]*viewVec[1];
	upVec[1] = rightVec[2]*viewVec[0] - rightVec[0]*viewVec[2];
	upVec[2] = rightVec[0]*viewVec[1] - rightVec[1]*viewVec[0];

	rightVec[0] *= X;
	rightVec[1] *= X;
	rightVec[2] *= X;
	upVec[0] *= Y;
	upVec[1] *= Y;
	upVec[2] *= Y;
	
	
	posX += rightVec[0] + upVec[0];
	posY += rightVec[1] + upVec[1];
	posZ += rightVec[2] + upVec[2];
	tarX += rightVec[0] + upVec[0];
	tarY += rightVec[1] + upVec[1];
	tarZ += rightVec[2] + upVec[2];
}

void Camera::CalculateCameraMatrix(float m[])
{
	float rightVec[3], upVec[3], viewVec[3];
	float norm;

	viewVec[0] = posX - tarX;
	viewVec[1] = posY - tarY;
	viewVec[2] = posZ - tarZ;
	norm = 1.0f/(float)sqrt(viewVec[0] * viewVec[0] + viewVec[1] * viewVec[1] + viewVec[2] * viewVec[2]);
	viewVec[0] *= norm;
	viewVec[1] *= norm;
	viewVec[2] *= norm;

	rightVec[0] = (float)(norY*viewVec[2] - norZ*viewVec[1]);
	rightVec[1] = (float)(norZ*viewVec[0] - norX*viewVec[2]);
	rightVec[2] = (float)(norX*viewVec[1] - norY*viewVec[0]);
	norm = 1.0f / (float)sqrt(rightVec[0] * rightVec[0] + rightVec[1] * rightVec[1] + rightVec[2] * rightVec[2]);
	rightVec[0] *= (float)norm;
	rightVec[1] *= (float)norm;
	rightVec[2] *= (float)norm;

	upVec[0] = (float)(viewVec[1]*rightVec[2] - viewVec[2]*rightVec[1]);
	upVec[1] = (float)(viewVec[2]*rightVec[0] - viewVec[0]*rightVec[2]);
	upVec[2] = (float)(viewVec[0]*rightVec[1] - viewVec[1]*rightVec[0]);

	m[ 0] = rightVec[0];	m[ 4] = rightVec[1];	m[ 8] = rightVec[2];	m[12] = - rightVec[0]*posX - rightVec[1]*posY - rightVec[2]*posZ;
	m[ 1] = upVec[0];		m[ 5] = upVec[1];		m[ 9] = upVec[2];		m[13] = - upVec[0]*posX - upVec[1]*posY - upVec[2]*posZ;
	m[ 2] = viewVec[0];		m[ 6] = viewVec[1];		m[10] = viewVec[2];		m[14] = - viewVec[0]*posX - viewVec[1]*posY - viewVec[2]*posZ;;
	m[ 3] = 0.0f;			m[ 7] = 0.0f;			m[11] = 0.0f;			m[15] = 1.0f;
}

void Camera::CalculateRightUp(float rightVec[], float upVec[])
{
	// 먼저 카메라의 view vector와 right vector를 알아낸다.
	float viewVec[3];
	float norm;

	viewVec[0] = posX - tarX;
	viewVec[1] = posY - tarY;
	viewVec[2] = posZ - tarZ;
	norm = 1/sqrt(viewVec[0] * viewVec[0] + viewVec[1] * viewVec[1] + viewVec[2] * viewVec[2]);
	viewVec[0] *= norm;
	viewVec[1] *= norm;
	viewVec[2] *= norm;
	
	rightVec[0] = (float)(norY*viewVec[2] - norZ*viewVec[1]);
	rightVec[1] = (float)(norZ*viewVec[0] - norX*viewVec[2]);
	rightVec[2] = (float)(norX*viewVec[1] - norY*viewVec[0]);
	norm = 1.0f / (float)sqrt(rightVec[0] * rightVec[0] + rightVec[1] * rightVec[1] + rightVec[2] * rightVec[2]);
	rightVec[0] *= (float)norm;
	rightVec[1] *= (float)norm;
	rightVec[2] *= (float)norm;

	upVec[0] = (float)(viewVec[1]*rightVec[2] - viewVec[2]*rightVec[1]);
	upVec[1] = (float)(viewVec[2]*rightVec[0] - viewVec[0]*rightVec[2]);
	upVec[2] = (float)(viewVec[0]*rightVec[1] - viewVec[1]*rightVec[0]);

}

void Camera::CalculateDistAngFromRaw()
{
	distance = (float)(sqrt((posX-tarX)*(posX-tarX) + (posY-tarY)*(posY-tarY) + (posZ-tarZ)*(posZ-tarZ)));
	angOrb = (float)(atan2(-(posZ-tarZ), posX-tarX)*180.0/PI);
	angLat = (float)(acos(sqrt((posX-tarX)*(posX-tarX) + (posZ-tarZ)*(posZ-tarZ))/distance)*180.0/PI);
}

void Camera::EditRotate(float orb, float lat, float lev)	// Camera 의 Rotation에 따른 조정
{
	angOrb += orb;
	angLat += lat;
	angLev += lev;
	
	if(angOrb >= 360.0)
		angOrb -= 360.0;
	if(angOrb < 0.0)
		angOrb += 360.0;
	
	if(angLat > 90.0)
		angLat = 90.0;
	if(angLat < -90.0)
		angLat = -90.0;

	posX = distance * (float)cos(RADIAN(angLat)) * (float)cos(RADIAN(angOrb)) + tarX;
	posY = distance * (float)sin(RADIAN(angLat)) + tarY;
	posZ = -(distance * (float)cos(RADIAN(angLat)) * (float)sin(RADIAN(angOrb)) - tarZ);
	norX = 1.0f * (float)cos(RADIAN(angLat+90.0)) * (float)cos(RADIAN(angOrb));
	norY = 1.0f * (float)sin(RADIAN(angLat+90.0));
	norZ = -1.0f * (float)cos(RADIAN(angLat+90.0)) * (float)sin(RADIAN(angOrb));
}

void Camera::EditOnFloor(float uptoFloor, float distEdit)
{
	// 카메라의 포지션이 floor 아래인 경우 수정해서 위로 올려준다.
	if(posY < uptoFloor && floorConstraint)
	{
		float lookVect[3], norm;
		float cameraPos[3], correctPos[3];
		lookVect[0] = tarX - posX;
		lookVect[1] = tarY - posY;
		lookVect[2] = tarZ - posZ;
		norm = 1.0f / distance;
		lookVect[0] *= norm;
		lookVect[1] *= norm;
		lookVect[2] *= norm;		

		// 걍 프로젝션한 경우
		norm = (uptoFloor - posY) / lookVect[1];
		cameraPos[0] = posX + norm*lookVect[0];
		cameraPos[1] = uptoFloor;
		cameraPos[2] = posZ + norm*lookVect[2];

		// 코렉션을 그해서 거리를 완벽하게 맞춘경우
		lookVect[0] = tarX - posX;
		lookVect[2] = tarZ - posZ;
		norm = 1.0f / (float)sqrt(lookVect[0]*lookVect[0] + lookVect[2]*lookVect[2]);
		lookVect[0] *= norm;
		lookVect[2] *= norm;
		norm = sqrt(distance*distance - (uptoFloor-tarY)*(uptoFloor-tarY));
		correctPos[0] = tarX - norm*lookVect[0];
		correctPos[1] = uptoFloor;
		correctPos[2] = tarZ - norm*lookVect[2];

		posX = distEdit*correctPos[0] + (1.0f-distEdit)*cameraPos[0];
		posY = distEdit*correctPos[1] + (1.0f-distEdit)*cameraPos[1];
		posZ = distEdit*correctPos[2] + (1.0f-distEdit)*cameraPos[2];
	}
}

void Camera::CalculateLookAt(float vec[3])
{
	float size;

	vec[0] = (float)(posX - tarX);
	vec[1] = (float)(posY - tarY);
	vec[2] = (float)(posZ - tarZ);
	size = (float)sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
	vec[0] /= size;
	vec[1] /= size;
	vec[2] /= size;
}

float Camera::getOrbit()
{
	return angOrb;
}

float Camera::getLevel()
{
	return angLat;
}

void Camera::GetRay(int width, int height, int x, int y, float *startpos, float *direction)
{
	float viewMat[3][3];
	CalculateLookAt(viewMat[2]);
	CalculateRightUp(viewMat[0], viewMat[1]);
	
	if(type == VIEW_PERS)
	{
		startpos[0] = (float)posX;
		startpos[1] = (float)posY;
		startpos[2] = (float)posZ;

		double n;
		float localDir[3], norm;
		n = (height / (2.0f*tan(RADIAN(0.5*fov))));
		localDir[2] = -(float)n;
		localDir[0] = (float)(x - 0.5*width);
		localDir[1] = -(float)(y - 0.5*height);
		norm = 1.0f / (float)sqrt(localDir[0]*localDir[0] + localDir[1]*localDir[1] + localDir[2]*localDir[2]);
		localDir[0] *= norm;
		localDir[1] *= norm;
		localDir[2] *= norm;

		direction[0] = viewMat[0][0]*localDir[0] + viewMat[1][0]*localDir[1] + viewMat[2][0]*localDir[2];
		direction[1] = viewMat[0][1]*localDir[0] + viewMat[1][1]*localDir[1] + viewMat[2][1]*localDir[2];
		direction[2] = viewMat[0][2]*localDir[0] + viewMat[1][2]*localDir[1] + viewMat[2][2]*localDir[2];
	}
	else
	{
		startpos[0] = (float)posX;
		startpos[1] = (float)posY;
		startpos[2] = (float)posZ;
	}
}