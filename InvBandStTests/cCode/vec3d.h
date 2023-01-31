#ifndef VEC3D_H
#define VEC3D_H
/*
* Header file for some simple 3d vector math, most (if not all) inlined
* for speed
*/
#include <math.h>

#ifdef __unix__
#define inline static __attribute__((always_inline))
#endif
#ifdef _WIN32
#define inline static __forceinline 
#endif



//Dot product for 3d vectors
inline double dot_vv(const double u[restrict 3], const double v[ [restrict 3]){
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}
//Norm of a 3d vector
inline double norm_v(const double v[[restrict 3]){
	return sqrt(dot_vv(v, v));
}

//Squared distance between two 3d vectors u, v 
inline double dist2_vv(const double u[[restrict  3], 
		       const double v[[restrict  3]){
	double ba; double res;
	ba = v[0] - u[0];
	res = ba*ba;
	ba = v[1] - u[1];
	res += ba*ba;
	ba = v[2] - u[2];
	return res + ba*ba;
};

//Squared distance between a 3d vector v and a point x, y, z
inline double dist2_vxyz(const double v[[restrict  3], 
						 const double x, const double y, const double z){
	double ba; double res;
	ba = v[0] - x;
	res = ba*ba;
	ba = v[1] - y;
	res += ba*ba;
	ba = v[2] - z;
	return res + ba*ba;
};

//Change of basis given matrix A whose mathematical columns are stored as rows
//Calculates x = Ay and stores the result in x
inline void cob(double x[[restrict 3], const double y[[restrict 3], 
		const double A[[restrict 3][[restrict 3]){
	///x = Ay where x are the new coordinates, A is the basis (a1 a2 a3)
	///i.e. they're column vectors, and y are the original coordinates
	///Here, A's basis is stored as row vectors, so the equation looks a bit 
	///different
	x[0] = A[0][0]*y[0] + A[1][0]*y[1] + A[2][0]*y[2];
	x[1] = A[0][1]*y[0] + A[1][1]*y[1] + A[2][1]*y[2];
	x[2] = A[0][2]*y[0] + A[1][2]*y[1] + A[2][2]*y[2];
}


#ifdef inline
#undef inline
#endif

#endif