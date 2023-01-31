#include <stdlib.h>
#include <string.h>

#include "General.h"
#include "Complex.h"
#include "Eigen.h"


//Aux function for calculating (a^2 + 1)^(1/2) w/o under/overflow
//If you aren't a coward, you'll turn off SAFE_QR and calculate it w/o
//any fancy manipulations
inline double shypot(const double a){
#if SAFE_QR
	double absa = fabs(a);
	if(absa > 1.0) return absa*sqrt(1.0 + 1.0/(absa*absa));
	return sqrt(1.0 + absa*absa);
#else
	return sqrt(a*a + 1.0);
#endif
}

//TODO: QRHERM CAN BE CLEANED UP A BIT:
//NOT TRANSPOSING A AND GETTING RID OF A FEW INITIAL VARIABLES
//NOT TRANSPOSING SAVES KEEPS EIGENVALS THE SAME, BUT EIGVECS MUCT BE CONJ'D

//QR Decomposition of a hermitian matrix into eigenvalues / eigenvectors
/* A: input matrix in 1-D form: A = {a00, a01, a10, a11} for a 2x2 matrix
** n:    dimension of A
** e:    pre-allocated array of eigenvalues, size n (no need to initialize)
** V:    pre-allocated ragged matrix of eigenvectors (vectors held row-wise)
** vecs: if true, will return the eigenvectors.  Otherwise, V will be garbage 
** sort: if true, eigenvals / vecs will be returned lowest to highest according
*        to the eigenvalue
*/
//Returns: 0 on success, 1 if the itr limit is reached
//         the ith entry of (e, V) is the ith eigenval, eigenvec pair.
//This is a re-implementation of an algorithm from 'mpmath':
//https://github.com/fredrik-johansson/mpmath
//which is itself based on a fortran implementation in 'EISPACK'.    
//This specific version has been edited to make the most accesses in row-order 
//instead of column order.
uint QRHerm(const uint n, const cdouble*restrict A, 
			double*restrict*restrict e, cdouble*restrict*restrict*restrict V,
			const uint vecs, const uint sort,
			const uint itrLim, const double eps){

	//Init (no, these can't be uints ... one specific part of the algo breaks
	//      if you try that ...)
	int i, j, k, l, m, w;
	int ii, ij, ji, il, ik, ki, jk, kj, iw, ip1w;
	const int N = (const int)n;
	uint itr;
	double sc, sci, f1, h, hh;
	double rtmp, r, s, c, p, b, f, dg;
	cdouble cg, ctmp, f0, accum;
	double* d = calloc(N, sizeof(double));
	cdouble* t = malloc(N*sizeof(cdouble));
	for(i = 0; i < N; ++i) cset(t + i, 0.0, 0.0);
	memset(*e, 0, N*sizeof(double)); ///evals -> zero for iterative updates 

	//Hermitian conjugate of A to make row-accesses correct
	cdouble* AT = malloc(N*N*sizeof(cdouble));
	for(i = 0; i < N; ++i){
		ii = acc(N, i, i);
		cset(AT + ii, A[ii].real, 0.0);
		for(j = i + 1; j < N; ++j){
			ij = acc(N, i, j);
			cset(AT + ij, A[ij].real, -A[ij].imag);
			cset(AT + acc(N, j, i), A[ij].real, A[ij].imag);
		}
	}

	// ...
	// ...
	// ...
	
	//--------------------------//
	//--- TRIDIAGONALIZATION ---//
	//--------------------------//
	cset(t + N - 1, 1.0, 0.0);
	for(i = N - 1; i > 0; --i){
		l = i - 1;

		///vector scaling
		sc = 0.0;
		for(j = 0; j < i; ++j){
			ij = acc(N, i, j);
			sc += fabs(AT[ij].real) + fabs(AT[ij].imag);
		}

		///skipping, stopping criteria
		if(fabs(sc) < eps){
			d[i] = 0.0;
			(*e)[i] = 0.0;
			cset(t + l, 1.0, 0.0);
			continue;
		}

		il = acc(N, i, l);
		if(i == 1){
			cset(&f0, AT[il].real, AT[il].imag);
			f1 = cfabs(f0);
			d[i] = f1;
			(*e)[i] = 0.0;
			if(f1 > eps){
				cmul_cc(&ctmp, t[i], f0); cdiv_cd(&ctmp, ctmp, f1);
				cset(t + l, ctmp.real, ctmp.imag);
			}
			else{
				cset(t + l, t[i].real, t[i].imag);
			}
			continue;
		}

		///setup for householder transformation
		sci = 1.0/sc; h = 0.0;
		for(j = 0; j < i; ++j){
			ij = acc(N, i, j);
			cmul_cd(AT + ij, AT[ij], sci);
			h += cfabs2(AT[ij]);
		}

		cset(&f0, AT[il].real, AT[il].imag);
		f1 = cfabs(f0); 
		cset(&cg, sqrt(h), 0.0);
		h += cg.real*f1;   ///at this point, g has no imaginary component
		d[i] = sc*cg.real; ///ditto ^
		if(f1 > eps){
			cdiv_cd(&f0, f0, f1);
			cmul_cc_n(&ctmp, f0, t[i]);
			cmul_cc(&cg, cg, f0);
		}
		else{
			ctmp.real = -t[i].real; ctmp.imag = -t[i].imag;
		}

		AT[il].real += cg.real; AT[il].imag += cg.imag;
		cset(&f0, 0.0, 0.0);

		///apply householder transformation
		for(j = 0; j < i; ++j){
			ij = acc(N, i, j);
			if(vecs) cdiv_cd(AT + acc(N, j, i), AT[ij], h);

			////AT dot U: expensive part + special eqtns = no function calls
			////(get over it)
			cset(&cg, 0.0, 0.0);
			for(k = 0; k < j + 1; ++k){
				/////g += AT(j, k)*  dot  AT(i, k)
				jk = acc(N, j, k); ik = acc(N, i, k);
				cg.real += AT[jk].real*AT[ik].real + AT[jk].imag*AT[ik].imag;
				cg.imag += AT[jk].real*AT[ik].imag - AT[jk].imag*AT[ik].real;
			}
			for(k = j + 1; k < i; ++k){
				/////g += AT(k, j)  dot  AT(i, k)
				kj = acc(N, k, j); ik = acc(N, i, k);
				cg.real += AT[kj].real*AT[ik].real - AT[kj].imag*AT[ik].imag;
				cg.imag += AT[kj].real*AT[ik].imag + AT[kj].imag*AT[ik].real;
			}

			////P (f0 += t[j]*  dot  A(i, j) 
			cdiv_cd(t + j, cg, h);
			f0.real += t[j].real*AT[ij].real + t[j].imag*AT[ij].imag;
			f0.imag += t[j].real*AT[ij].imag - t[j].imag*AT[ij].real;
		}

		///reduce AT, get Q
		hh = 0.5*f0.real/h; ////by now, f0 is real (to numerical precision)
		for(j = 0; j < i; ++j){
			ij = acc(N, i, j);
			cset(&f0, AT[ij].real, AT[ij].imag);
			cg.real = t[j].real - hh*f0.real; cg.imag = t[j].imag - hh*f0.imag;
			cset(t + j, cg.real, cg.imag);

			////expensive part + complicated eqation = no functions
			////again, get over it
			for(k = 0; k < j + 1; k++){
				/////AT(j, k) -= f0*  dot  t[k]    +    g*  dot  AT(i, k)
				jk = acc(N, j, k); ik = acc(N, i, k);
				AT[jk].real -= f0.real*t[k].real + f0.imag*t[k].imag + 
							   cg.real*AT[ik].real + cg.imag*AT[ik].imag;
				AT[jk].imag -= f0.real*t[k].imag - f0.imag*t[k].real +
							   cg.real*AT[ik].imag - cg.imag*AT[ik].real;
			}
		}

		cset(t + l, ctmp.real, ctmp.imag);
		(*e)[i] = h;
	}

	// ...
	// ...
	// ...

	//--------------------------//
	//------ INTERMISSION ------//
	//--------------------------//
	//Convient to shift off-diagonals by 1
	for(i = 1; i < N; ++i) d[i - 1] = d[i];
	d[N - 1] = 0.0;
	//Also need to update current accum eigs and AT
	(*e)[0] = 0.0;
	for(i = 0; i < N; ++i){
		ii = acc(N, i, i);
		rtmp = (*e)[i];
		(*e)[i] = AT[ii].real;
		AT[ii].real = rtmp;
	}

	//Finally, set V to identity if we want the eigenvectors
	if(vecs){
		for(i = 0; i < N; ++i){
			cset(&((*V)[i][i]), 1.0, 0.0);
			for(j = i + 1; j < N; ++j){
				cset(&((*V)[i][j]), 0.0, 0.0);
				cset(&((*V)[j][i]), 0.0, 0.0);
			}
		}
	}

	// ...
	// ...
	// ...

	//--------------------------//
	//--- EIGENVALUES / VECS ---//
	//--------------------------//
	for(l = 0; l < N; ++l){
		itr = 0;

		while(1){
			///Grab a small off-diag element
			m = l;
			while(1){
				if(m + 1 == N) break;
				if(fabs(d[m]) < eps*(fabs((*e)[m]) + fabs((*e)[m+1]))) break;
				m += 1;
			}
			if(m == l) break;
	
			if(itr >= itrLim) return 1u;  ////prevent hanging if QR fails		
			itr++;

			///shift
			dg = 0.5*((*e)[l+1] - (*e)[l])/d[l];
			r = shypot(dg);
			s = (dg < 0.0)? dg - r : dg + r;
			dg = (*e)[m] - (*e)[l] + d[l]/s;

			///plane->Givens rotations: get back to tridiagonal form
			s = 1.0; c = 1.0; p = 0.0;
			for(i = m - 1; i > l - 1; --i){
				f = s*d[i]; b = c*d[i];

				////improves convergence by choosing a large denom
				if(fabs(f) > fabs(dg)){
					c = dg/f;
					r = shypot(c);
					d[i + 1] = f*r;
					s = 1.0/r;
					c *= s;
				}
				else{
					s = f/dg;
					r = shypot(s);
					d[i + 1] = dg*r;
					c = 1.0/r;
					s *= c;
				}

				dg = (*e)[i + 1] - p;
				r = ((*e)[i] - dg)*s + 2.0*c*b;
				p = s*r;
				(*e)[i + 1] = dg + p;
				dg = c*r - b;

				////accum eigenvectors only if we want them
				///At this point, we can deal with all real arithmatic
				///even though V will eventually have imaginary components
				if(vecs){
					for(w = 0; w < N; ++w){
						f = (*V)[i+1][w].real;
						////V(i+1, w) = s*V(i, w) + c*f
						(*V)[i+1][w].real = s*(*V)[i][w].real + c*f;
						////V(i, w) = c*V(i, w) - s*f
						(*V)[i][w].real = c*(*V)[i][w].real - s*f;
					}
				}
			}

			///finish up updating
			(*e)[l] -= p;
			d[l] = dg;
			d[m] = 0.0;
		}
	}

	// ...
	// ...
	// ...

	//--------------------------//
	//---- FINISH EIGENVECS ----//
	//--------------------------//
	//Grab the imaginary components for the eigenvectors
	if(vecs){
		for(i = 0; i < N; ++i){
			for(j = 0; j < N; ++j){
				ij = acc(N, i, j);
				cmul_cc(&((*V)[i][j]), (*V)[i][j], t[j]);
			}
		}
		for(i = 0; i < N; ++i){
			if(fabs(AT[acc(N, i, i)].real) < eps) continue;
			for(j = 0; j < N; ++j){
				cset(&ctmp, 0.0, 0.0);
				///Expensive part of the algo + complicated math = no functions
				///last time, I promise :)
				for(k = 0; k < i; ++k){
					////ctmp += AT(k, i)*  dot  V(j, k)
					ki = acc(N, k, i);
					ctmp.real += AT[ki].real*(*V)[j][k].real +
								 AT[ki].imag*(*V)[j][k].imag;
					ctmp.imag += AT[ki].real*(*V)[j][k].imag -
								 AT[ki].imag*(*V)[j][k].real;
				}
				for(k = 0; k < i; ++k){
					////V(j, k) -= ctmp  dot  A(i, k)
					ik = acc(N, i, k);
					(*V)[j][k].real -= ctmp.real*AT[ik].real - 
									   ctmp.imag*AT[ik].imag;
					(*V)[j][k].imag -= ctmp.real*AT[ik].imag +
									   ctmp.imag*AT[ik].real;
				}
			}
		}
	}

	// ...
	// ...
	// ...

	//--------------------------//
	//---- SORT EIGVAL/VECS ----//
	//--------------------------//
	//Simple insertion sort - unlikley to be responsible for slowdowns, but 
	//if it is, consider doing a quicksort / insertion sort hybrid
	if(sort){
		if(vecs){
			double* ea_; double* eb_; double et_;
			cdouble** va_; cdouble** vb_; cdouble* vt_;
			for(i = 0; i < N; ++i){
				for(j = i; j > 0;){
						eb_ = *e + j;
						vb_ = *V + j;
						ea_ = *e + --j;
						va_ = *V + j;
					if(*eb_ < *ea_){
						et_ = *ea_;
						*ea_ = *eb_;
						*eb_ = et_;
						vt_ = *va_;
						*va_ = *vb_;
						*vb_ = vt_;
					}
				}
			}
		}
		else{
			double* ea_; double* eb_; double et_;
			for(i = 0; i < N; ++i){
				for(j = i; j > 0;){			
					eb_ = *e + j;
					ea_ = *e + --j;
					if(*eb_ < *ea_){
						et_ = *ea_;
						*ea_ = *eb_;
						*eb_ = et_;
					}
				}
			}
		}
	}


	
	free(d); free(t); free(AT);
	return 0u;
}


