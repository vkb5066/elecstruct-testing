#ifndef GENERAL_H
#define GENERAL_H

#ifdef __unix__
#define inline static __attribute__((always_inline)) //i didn't ask >:(
#endif
#ifdef _WIN32
#define restrict __restrict 
#define inline static __forceinline //i didn't ask >:(
#endif

//Helpful things
#define NOP ;
#define min(X,Y) (((X) < (Y)) ? (X) : (Y))
#define max(X,Y) (((X) > (Y)) ? (X) : (Y))
#define acc(N, I, J) ((N)*(I) + (J))

//Redefitions
typedef unsigned int uint;

//Physics, math constants
#define TWOPI              6.283185307
#define HBAR_SQ_OVER_TWO_M 3.809981746 ///eV A^2

//QR algorithm tuning
#define SAFE_QR 1u
#define QR_ITRLIM 30u
#define QR_EPS 1.0e-8

#endif