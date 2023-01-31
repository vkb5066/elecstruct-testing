#ifndef EIGEN_H
#define EIGEN_H 

#include "General.h"
#include "Complex.h"

uint QRHerm(const uint n, const cdouble*restrict A, 
			double*restrict*restrict e, cdouble*restrict*restrict*restrict V,
			const uint vecs, const uint sort,
			const uint itrLim, const double eps);

#endif