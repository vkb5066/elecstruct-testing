#ifndef ATOMSTRUCT_H
#define ATOMSTRUCT_H
/*
* Header file for the atom structures involved in ionic relaxations 
*/
#include "vec3d.h"

#ifdef __unix__
#define restrict restrict 
#define inline static __attribute__((always_inline))
#endif
#ifdef _WIN32
#define restrict __restrict 
#define inline static __forceinline 
#endif

typedef struct atom atom;
struct atom{
	unsigned int* spec; ///id number corresponding to an element (or vacancy)
	double crdsf[3]; ///fractional (crystal) coordinates a, b, c 
	double crdsc[3]; ///cartesian coordinates x, y, z

	unsigned int npairs;
	atom** pairs; ///pointers to this atom's neighbors
	unsigned int* ntrips;
	atom*** trips; ///trips[i][j] = ith pair's jth neighbor s.t. 
				   ///dists(this atom, ith pair, jth pair) all < cutoff rad

	unsigned int nimgs;
	atom** imgs; ///images of this atom near, but not in, bounding region
				 ///should only be set for the atoms within sim region

	atom* self;
};

//Set cartesian coordinates of an atom given the basis (stored as row vectors)
inline void carcrds_a(atom* a, const double A[restrict 3][restrict 3]){
	///c = Ad where c are the cartesian coordinates, A is the basis (a1 a2 a3)
	///i.e. they're column vectors, and d are the direct coordinates
	cob(&((*a).crdsc), (*a).crdsf, A);
}
//Set fractional coordinates of atom given basis inv (stored as row vectors)
inline void fracrds_a(atom* a, const double Ai[restrict 3][restrict 3]){
	///d = A^(-1)c where c are the cartesian coordinates, A^(-1) is the basis 
	///inverse (a1 a2 a3) as column vectors, and d are the direct coordinates
	cob(&((*a).crdsf), (*a).crdsc, Ai);
}

//Sets all atom pairs and triplets
//Returns 1 on error
int setnbrs_a(const unsigned int natoms, atom*restrict*restrict atoms,
			  unsigned int*restrict nxatoms, atom*restrict*restrict xatoms,
			  const double A[restrict 3][restrict 3], 
			  const double rcut, const double acut, const double eps);

//Completly frees the atoms arras that were set by BOTH latticeparse() and 
//setnbrs_a().  Use at end of program, not for resetting neighbor counts
void free_a(const unsigned int natoms, atom*restrict*restrict atoms,
			atom*restrict*restrict xatoms);

//Shifts an atom and its images' cartesian coordinates 
//as xnew = xold + delta (delta = (dx, dy, dz))
inline void carshift_a(atom* a, const double delta[3]){
	(*a).crdsc[0] += delta[0];
	(*a).crdsc[1] += delta[1];
	(*a).crdsc[2] += delta[2];
	for(unsigned int i = 0u; i < (*a).nimgs; ++i){
		(*a).imgs[i]->crdsc[0] += delta[0];
		(*a).imgs[i]->crdsc[1] += delta[1];
		(*a).imgs[i]->crdsc[2] += delta[2];
	}
}

#ifdef restrict
#undef restrict
#endif
#ifdef inline
#undef inline
#endif

#endif