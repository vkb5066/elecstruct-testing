#include <stdlib.h>

#include "General.h"
#include "Complex.h"
#include "Hamiltonian.h"

double rfloat(const double mini, const double maxi){
	return mini + ((maxi - mini)*rand()/RAND_MAX);
}

int main(int argc, char* argv[]){



	// --- G Vector Finding Testing ---
	double B[3u][3u] = {{2.0, 0.0, 0.0}, 
						{0.0, 2.0, 0.0}, 
						{0.0, 0.0, 2.0}};
	double encut = 500.0;
	uint nGVecs; rlvec* gVecs;
	FetchGVecs(&nGVecs, &gVecs, encut, B);



	
	// --- Matrix Diagonalization Testing ---
	srand(15u);

	uint dim = 125;
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

	uint nSize = dim*(dim+1u)/2u; cdouble* symmat = malloc(nSize*sizeof(cdouble));
	MatrixTransform(dim, mat, &symmat);
	JacobiEigenDecomp(dim, &symmat, MAX_JACOBI_STEPS, JACOBI_SKIP_TOL, JACOBI_TRACE_TOL);


	return 0;
}