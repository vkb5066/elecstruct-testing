#include <stdlib.h>
#include <stdio.h>
#include "General.h"
#include "Complex.h"
#include "Eigen.h"

double rfloat(const double mini, const double maxi){
	return mini + ((maxi - mini)*rand()/RAND_MAX);
}

int main(int argc, char* argv[]){



	
	// --- Matrix Diagonalization Testing ---
	srand(150u);

	uint dim = 30;
	cdouble** mat = malloc(dim*sizeof(cdouble*));
	for(uint i = 0; i < dim; ++i){
		mat[i] = malloc(dim*sizeof(cdouble));
	}
	
	double min_ = -10.0; double max_ = +10.0;
	for(uint i = 0; i < dim; ++i){
		mat[i][i].real = rfloat(min_, max_);
		mat[i][i].imag = 0.0;
		for(uint j = i + 1u; j < dim; ++j){
			mat[i][j].real = rfloat(min_, max_);
			mat[i][j].imag = rfloat(min_, max_);
			mat[j][i].real = mat[i][j].real;
			mat[j][i].imag = -mat[i][j].imag;
		}
	}
	
	
	cdouble* H = malloc(dim*dim*sizeof(cdouble));
	for(uint i = 0; i < dim; ++i){
		for(uint j = 0; j < dim; ++j){
			H[acc(dim, i, j)].real = mat[i][j].real;
			H[acc(dim, i, j)].imag = mat[i][j].imag;
		}
	}

	double* eigenVals = malloc(dim*sizeof(double));
	cdouble** eigenVecs = malloc(dim*sizeof(cdouble*));
	for(uint i = 0; i < dim; ++i){
		eigenVecs[i] = malloc(dim*sizeof(cdouble));
	}
	uint state = QRHerm(dim, H, &eigenVals, &eigenVecs, 1u, 1u, QR_ITRLIM, QR_EPS);


	//Check A dot v - lambda dot v ~ 0



	return 0;
}