#ifndef GENERAL_H
#define GENERAL_H

#ifdef _WIN32
#define restrict __restrict 
#endif

//Helpful things
#define NOP ;
#define min(X,Y) (((X) < (Y)) ? (X) : (Y))
#define max(X,Y) (((X) > (Y)) ? (X) : (Y))

//Redefitions
typedef unsigned int uint;

//Physics, math constants
#define TWOPI              6.283185307
#define HBAR_SQ_OVER_TWO_M 3.809981746 ///eV A^2

//Other constants
#define JACOBI_SKIP_TOL 1.0e-3 ///tol before off-diag entry is zero
#define JACOBI_TRACE_TOL 1.0e-3 ///energy diff before convergence reached
#define MAX_JACOBI_STEPS 15u ///number of steps before jacobi algo gives up

#endif