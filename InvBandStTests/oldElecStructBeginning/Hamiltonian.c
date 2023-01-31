#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "General.h"
#include "Vec3d.h"
#include "Hamiltonian.h"
#ifdef __unix__
#include "extern/kdtree-master/kdtree.h"
#endif
#ifdef _WIN32
#include "extern//kdtree-master//kdtree.h"
#endif


//Fills g* with the values corresponding to 
// (gx, gy, gz) = h*b1 + k*b2 + l*b3
//Note that B[3][3] is stored as {b1, b2, b3} in row-order i.e.
// (b1_x, b1_y, b1_z) = (B[0][0], B[0][1], B[0][2])
void SetGVecPts(double*restrict gx, double*restrict gy, double*restrict gz,
				const int h, const int k, const int l,
				const double B[3u][3u]){
	*gx = (double)h*B[0][0] + (double)k*B[1][0] + (double)l*B[2][0];
	*gy = (double)h*B[0][1] + (double)k*B[1][1] + (double)l*B[2][1];
	*gz = (double)h*B[0][2] + (double)k*B[1][2] + (double)l*B[2][2];
}


//Returns the array of all G vectors within a sphere corresponding to the
//cutoff energy
void FetchGVecs(uint*restrict nGVecs, rlvec*restrict*restrict rlVecs, 
				const double enCut, const double B[3u][3u], 
				const double kpt[3u]){

	double gCut = sqrt(1/HBAR_SQ_OVER_TWO_M*enCut);
	int hklMaxes[3u] = {(int)(gCut / sqrt(Mag2(B[0]))), ///max number for h
						(int)(gCut / sqrt(Mag2(B[1]))), ///max search for k
						(int)(gCut / sqrt(Mag2(B[2])))};///max search for l

	//Make ALL g vectors within the needed search radius, p
	double gx, gy, gz;
	void* kdNode = kd_create(3);
	for(int h = -hklMaxes[0]; h <= hklMaxes[0]; ++h){
	for(int k = -hklMaxes[1]; k <= hklMaxes[1]; ++k){
	for(int l = -hklMaxes[2]; l <= hklMaxes[2]; ++l){
		SetGVecPts(&gx, &gy, &gz, h, k, l, B);
		gx += kpt[0]; gy += kpt[1]; gz += kpt[2];
		kd_insert3(kdNode, gx, gy, gz, NULL); ///actually inserting k + G
	}
	}
	}
	
	//Only keep the ones within GCut
	///Grab KD tree results ...
	double zero = 0.0; ////necessary to stop crashes on unix lol
	struct kdres* res = kd_nearest_range3(kdNode, zero, zero, zero, gCut);
	uint resSize = (uint)kd_res_size(res);

	*nGVecs = resSize;
	*rlVecs = malloc(resSize*sizeof(rlvec)); 
	/// ... and make G vectors out of them
	/// (dont forget to transform from k + G back to G)
	for(uint i = 0; i < resSize; ++i){
		kd_res_item3(res, &gx, &gy, &gz);
		(*rlVecs)[i].crdsC[0] = gx - kpt[0];
		(*rlVecs)[i].crdsC[1] = gy - kpt[1];
		(*rlVecs)[i].crdsC[2] = gz - kpt[2];
		(*rlVecs)[i].id = i;
		kd_res_next(res);
	}
	kd_res_free(res);
}


//Transforms the (0-based) indices to a normal matrix M[i][j] -> the index
//to the symmetric 1D array
//size refers to the N in describing the matrix M as size NxN
uint IndexTransform(const uint i, const uint j, const uint size) {
	if (i <= j)
		return (2u*size - i - 1u)*i/2u + j;
	return (2u*size - j - 1u)*j/2u + i;
}

//Transforms a 2d hermitian matrix into a 1d symmetric array
//the order is:
/*       /         \
*       |  0  1  2  |
* Mat = |  1* 3  4  |  --> Array = {0 1 2 3 4 5}
*       |  2* 4* 5  |
*        \         /
*/ 
void MatrixTransform(const uint size, const cdouble*restrict*restrict mat,
					 cdouble*restrict*restrict arr){
	uint counter = 0;
	for(uint i = 0; i < size; ++i){
		for(uint j = i; j < size; ++j){
			(*arr)[counter].real = mat[i][j].real;
			(*arr)[counter].imag = mat[i][j].imag;
			counter++;
		}
	}
}

//Computes the trace of the absolute values of arr divided by the array size
double AvgTraceAbs(const uint size, cdouble* arr){
	double avg = 0.0;
	for(uint i = 0; i < size; ++i){
		avg += fabs(arr[IndexTransform(i, i, size)].real);
	}

	return avg/(double)size;
}

//Uses a jacobi decomposition based on 
//"Matrix Computations" - 3rd ed. by Gene H. Golub
//and edited to work on hermitian matrices
//size is the dimensions of A, and arr is A in condensed 1D form
//tol is the difference in average of diagonals between two sweeps before the 
//program terminates

///ab + cd where a is real-valued
void AuxMultiply1(cdouble* res, const double a, const cdouble b,
				  const cdouble c, const cdouble d){
	res->real = a*b.real + c.real*d.real - c.imag*d.imag;
	res->imag = a*b.imag + c.real*d.imag + c.imag*d.real;
}
///ab - cd where a is real-valued
void AuxMultiply2(cdouble* res, const double a, const cdouble b,
				  const cdouble c, const cdouble d){
	res->real = a*b.real - c.real*d.real + c.imag*d.imag;
	res->imag = a*b.imag - c.real*d.imag - c.imag*d.real;
}
///ab + cd* where a is real-valued
void AuxMultiply3(cdouble* res, const double a, const cdouble b,
				  const cdouble c, const cdouble d){
	res->real = a*b.real + c.real*d.real + c.imag*d.imag;
	res->imag = a*b.imag - c.real*d.imag + c.imag*d.real;
}
///ab - cd* where a is real-valued
void AuxMultiply4(cdouble* res, const double a, const cdouble b,
				  const cdouble c, const cdouble d){
	res->real = a*b.real - c.real*d.real - c.imag*d.imag;
	res->imag = a*b.imag + c.real*d.imag - c.imag*d.real;
}

void JacobiEigenDecomp(const uint size, cdouble*restrict*restrict arr, 
					   const uint maxSweeps, 
					   const double skipTol, const double itrTol){
	uint ii, ij, jj;
	uint ki, kj, ik, jk;
	double t, c, s, sign; cdouble sp, sm;
	double absiijj; double absij;
	cdouble tmp;
	double traceOld, traceCur, dt;

	printf("n        <Tr(|H|)>        d<Tr(|H|)>/dn\n");
	printf("---------------------------------------\n");
	for(uint sweepInd = 0; sweepInd < maxSweeps; ++sweepInd){

		for(uint i = 0; i < size - 1u; ++i){
		ii = IndexTransform(i, i, size);
		for(uint j = i + 1u; j < size; ++j){
			///No need for expensive calculations if this entry is near 0
			ij = IndexTransform(i, j, size);
			if(cabs2((*arr)[ij]) < skipTol*skipTol) continue;
			jj = IndexTransform(j, j, size);

			///Setup the rest of the i, j components
			sign = ((*arr)[ii].real - (*arr)[jj].real > 0.0) ? +1.0 : -1.0; 
			absij = cabs((*arr)[ij]);
			absiijj = fabs((*arr)[ii].real - (*arr)[jj].real);
			t = 2.0*cabs((*arr)[ij])*sign /
				(  absiijj + sqrt(absiijj*absiijj + 4.0*absij*absij)  );
			c = 1.0/sqrt(1.0 + t*t);
			s = t*c;
			cmul_cd(&sp, (*arr)[ij], s/absij);
			cdiv_dc(&sm, s*absij, (*arr)[ij]);

			///Apply rotation matrices
			(*arr)[ii].real += t*absij;
			(*arr)[jj].real -= t*absij;
			(*arr)[ij].real = 0.0; (*arr)[ij].imag = 0.0;

			for(uint k = 0; k < i; ++k){
				ki = IndexTransform(k, i, size);
				kj = IndexTransform(k, j, size);
				AuxMultiply1(&tmp, c, (*arr)[ki], sm, (*arr)[kj]);
				AuxMultiply2(&(*arr)[kj], c, (*arr)[kj], sp, (*arr)[ki]);
				(*arr)[ki].real = tmp.real; (*arr)[ki].imag = tmp.imag;
			}
			for(uint k = i + 1u; k < j; ++k){
				ik = IndexTransform(i, k, size);
				kj = IndexTransform(k, j, size);
				AuxMultiply3(&tmp, c, (*arr)[ik], sp, (*arr)[kj]);
				AuxMultiply4(&(*arr)[kj], c, (*arr)[kj], sp, (*arr)[ik]);
				(*arr)[ik].real = tmp.real; (*arr)[ik].imag = tmp.imag;
			}
			for(uint k = j + 1u; k < size; ++k){
				ik = IndexTransform(i, k, size);
				jk = IndexTransform(k, j, size);
				AuxMultiply1(&tmp, c, (*arr)[ik], sp, (*arr)[jk]);
				AuxMultiply2(&(*arr)[jk], c, (*arr)[jk], sm, (*arr)[ik]);
				(*arr)[ik].real = tmp.real; (*arr)[ik].imag = tmp.imag;
			}
		}
		}

		//Print out converg info ...
		traceCur = AvgTraceAbs(size, (*arr));
		dt = (sweepInd == 0u) ? traceCur : fabs(traceCur - traceOld);
		
		printf("%02u / %02u  %014.10f  %014.10f\n", 
			   sweepInd + 1u, maxSweeps, traceCur, dt);
		// ... and exit if we've converged
		if(dt < itrTol){
			return;
		}
		traceOld = traceCur;
	}

	//If we're here, we haven't reached convergence
	printf("WARNING: eigenval convergence not reached!\n");
}