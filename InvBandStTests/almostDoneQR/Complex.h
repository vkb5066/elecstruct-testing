#ifndef COMPLEX_H
#define COMPLEX_H 

#include <math.h>

#include "General.h"

typedef struct complex_double cdouble;
struct complex_double{
	double real;
	double imag;
};

//Cleanly updates both the real and imaginary portions of z
inline void cset(cdouble* res, const double r, const double i){
	res->real = r;
	res->imag = i;
}

//Returns the absolute value of z: sqrt( re(z)^2 + im(z)^2 )
inline double cfabs(const cdouble z){
	return sqrt(z.real*z.real + z.imag*z.imag);
}

//Returns the absolute value of z, squared
inline double cfabs2(const cdouble z){
	return z.real*z.real + z.imag*z.imag;
}

//Multiplication of a complex by a real
inline void cmul_cd(cdouble* res, const cdouble a, const double b){
	res->real = a.real*b;
	res->imag = a.imag*b;
}

//Multiplication of two complex vals
inline void cmul_cc(cdouble* res, const cdouble a, const cdouble b){
	res->real = a.real*b.real - a.imag*b.imag;
	res->imag = a.real*b.imag + a.imag*b.real;
}

//Multiplication of two complex vals, one negative
inline void cmul_cc_n(cdouble* res, const cdouble pa, const cdouble nb){
	res->real = -pa.real*nb.real + pa.imag*nb.imag;
	res->imag = -pa.real*nb.imag - pa.imag*nb.real;
}

//Division of a complex by a real
inline void cdiv_cd(cdouble* res, const cdouble num, const double den){
	res->real = num.real/den;
	res->imag = num.imag/den;
}
//Division of a real by a complex
inline void cdiv_dc(cdouble* res, const double num, const cdouble den){
	double denom = den.real*den.real + den.imag*den.imag;
	res->real =  num*den.real/denom;
	res->imag = -num*den.imag/denom;
}
//Division of two complex vals
inline void cdiv_cc(cdouble* res, const cdouble num, const cdouble den){
	double denom = den.real*den.real + den.imag*den.imag;
	res->real = (num.real*den.real + num.imag*den.imag) / denom;
	res->imag = (num.imag*den.real - num.real*den.imag) / denom;
}

#endif