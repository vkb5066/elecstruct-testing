#ifndef HAMIL_H
#define HAMIL_H

#include "General.h"
#include "Complex.h"

typedef struct rlvec rlvec;
struct rlvec{
	uint id;
	double crdsC[3u];
};
void SetGVecPts(double*restrict gx, double*restrict gy, double*restrict gz,
				const int h, const int k, const int l,
				const double B[3u][3u]);
void FetchGVecs(uint*restrict nGVecs, rlvec*restrict*restrict rlVecs, 
				const double enCut, const double B[3u][3u], 
				const double kpt[3u]);


uint IndexTransform(const uint i, const uint j, const uint size);
void MatrixTransform(const uint size, const cdouble*restrict*restrict mat,
					 cdouble*restrict*restrict arr);

double AvgTraceAbs(const uint size, cdouble* arr);
void JacobiEigenDecomp(const uint size, cdouble*restrict*restrict arr, 
					   const uint maxSweeps, double skipTol, double itrTol);



#endif