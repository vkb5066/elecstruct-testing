#include <stdlib.h>

#include "atomstruct.h"
#include "vec3d.h"

#ifdef __unix__
#include "extern/kdtree-master/kdtree.h"
#endif
#ifdef _WIN32
#include "extern//kdtree-master//kdtree.h"
#endif

#ifdef __unix__
#define restrict restrict 
#define inline static __attribute__((always_inline))
#endif
#ifdef _WIN32
#define restrict __restrict 
#define inline static __forceinline 
#endif



//Sets pairs, trips, and periodic images for a set of base atoms
//allocates extra atoms (the periodic images) and stores them in xatoms
//If using this to RESET, the ALL of xatoms as well as the images, pairs,
//and trips of base atoms must be freed beforehand
//rcut is the cutoff distance between a central atom and a direct pair
//acut is the angle cutoff for a triplet (radians, as god intended), [0, pi]
#define IS_IMAGE UINT_MAX
#define IS_IMPORTANT_IMAGE UINT_MAX - 1u
int setnbrs_a(const unsigned int natoms, atom*restrict*restrict atoms,
			  unsigned int*restrict nxatoms, atom*restrict*restrict xatoms,
			  const double A[restrict 3][restrict 3], 
			  const double rcut, const double acut, const double eps){
	//Counters
	unsigned int i, j, k, m, ii;
	int a, b, c;
	//Number of images
	unsigned int nimgstot; int nimgs[3u];
	double latlen;
	//Periodic image setting
	unsigned int nallatoms, reset, caa, cta;
	atom* allatoms; ///!dyn mem
	//Kd Tree
	void* kdnode; ///!dyn mem
	struct kdres* res; ///!dyn mem
	//Pair / trip setting
	int ressize;
	double x, y, z, cosjik, cosacut;
	double ij[3], nij, ik[3]; ///rj-ri, |rj-ri|, rk-ri


	//Find the number of periodic images needed to satisfy rcut
	for(i = 0u; i < 3u; ++i){
		latlen = sqrt(A[i][0]*A[i][0] + A[i][1]*A[i][1] + A[i][2]*A[i][2]);
		for(j = 1u;; ++j){
			if(rcut < (double)j*latlen){
				nimgs[i] = j;
				break;
			}
		}
	}
	nimgstot = (2u*nimgs[0]+1u)*(2u*nimgs[1]+1u)*(2u*nimgs[2]+1u);


	//Periodic image loop
	nallatoms = nimgstot*natoms;
	allatoms = malloc(nallatoms*sizeof(atom));
	for(i = 0u, caa = 0u; i < natoms; ++i){
		///Ensure that this atom is in the cell
		reset = 0u;
		for(j = 0u; j < 3u; ++j){
			while((*atoms)[i].crdsf[j] < 0.0){
				(*atoms)[i].crdsf[j] += 1.0;
				reset = 1u;
			}
			while((*atoms)[i].crdsf[j] >= 1.0){
				(*atoms)[i].crdsf[j] -= 1.0;
				reset = 1u;
			}
		}
		if(reset) carcrds_a(atoms + i, A);

		///Explicitly deal with a=b=c=0: this is an atom inside the cell
		///and must keep track of its periodic images
		////Deepcopy important info from 'atoms' ... ii indexes this atom
		ii = (nimgstot)*(i+1u) - 1u; 
		allatoms[ii].spec = (*atoms)[i].spec;
		for(j = 0u; j < 3u; ++j){
			allatoms[ii].crdsf[j] = (*atoms)[i].crdsf[j];
			allatoms[ii].crdsc[j] = (*atoms)[i].crdsc[j];
		}
		allatoms[ii].nimgs = nimgstot - 1u; ////minus the 0, 0, 0 "image"
		allatoms[ii].imgs = malloc((allatoms[ii].nimgs)*sizeof(atom*));
		allatoms[ii].self = &(*atoms)[i];

		cta = 0u;
		for(a = -nimgs[0]; a <= nimgs[0]; ++a)
		for(b = -nimgs[1]; b <= nimgs[1]; ++b)
		for(c = -nimgs[2]; c <= nimgs[2]; ++c){
			if(!a && !b && !c) continue;

			////Deepcopy important info from 'atoms' (+ coord shift)
			///but keep a reference to the original species
			///(we want to independently change coords, but always keep the
			///atom species of images identical for atom hopping later)
			allatoms[caa].spec = allatoms[ii].spec;
			allatoms[caa].crdsf[0] = (*atoms)[i].crdsf[0] + (double)a;
			allatoms[caa].crdsf[1] = (*atoms)[i].crdsf[1] + (double)b;
			allatoms[caa].crdsf[2] = (*atoms)[i].crdsf[2] + (double)c;
			carcrds_a(&allatoms[caa], A);
			allatoms[caa].nimgs = IS_IMAGE; ////use this entry as a flag
			
			////Add this shifted atom to the periodic images
			allatoms[ii].imgs[cta] = &allatoms[caa];

			cta++;
			caa++;
		}
		
		///Now that we have all of the periodic images set, we can add this
		///atom to the array of all atoms
		caa++;
	}


	//Fill in a KD tree for quick nearest neighbor searches
	kdnode = kd_create(3);
	for(i = 0u; i < nallatoms; ++i){
		kd_insert3(kdnode, allatoms[i].crdsc[0], allatoms[i].crdsc[1], 
				   allatoms[i].crdsc[2], i);
	}


	//Neigbor setting loop (pairs, trips)
	//The looping indices guarentee that we only query atoms in the original 
	//cell
	//At the same time, we also reduce the number of images of each base atom
	//from the max possible number down to the minimum number
	*nxatoms = 0u; *xatoms = malloc(nallatoms*sizeof(atom));
	cosacut = cos(acut);
	for(i = nimgstot-1u, ii = 0u; i < nallatoms; i += nimgstot, ++ii){

		///Query the kd tree
		x = allatoms[i].crdsc[0];
		y = allatoms[i].crdsc[1];
		z = allatoms[i].crdsc[2];
		res = kd_nearest_range3(kdnode, x, y, z, rcut);
		ressize = kd_res_size(res);

		///--- Set Pairs ------------------------------------------------------
		allatoms[i].npairs = (unsigned int)ressize - 1u;
		(*atoms)[ii].npairs = allatoms[i].npairs;
		allatoms[i].pairs = malloc(allatoms[i].npairs*sizeof(atom*));
		(*atoms)[ii].pairs = malloc((*atoms)[ii].npairs*sizeof(atom*));

		for(j = 0u, k = 0u; j < (unsigned int)ressize; ++j){
			cta = (unsigned int)kd_res_item3(res, &x, &y, &z);
			if(dist2_vxyz(allatoms[i].crdsc, x, y, z) < eps) goto skip;

			////Another flag - this one for showing that this atom is either
			////a base atom or the neighbor of one
			////But only set if it isn't already set to a reasonable
			////value
			if(allatoms[cta].nimgs == IS_IMAGE){
				allatoms[cta].nimgs = IS_IMPORTANT_IMAGE;
				/////here, this is a unique periodic image.  we want to keep
				/////these after function term, so deepcopy it into the return
				////extras array
				(*xatoms)[*nxatoms].spec = allatoms[cta].spec;
				for(m = 0u; m < 3u; ++m){
					(*xatoms)[*nxatoms].crdsf[m] = allatoms[cta].crdsf[m];
					(*xatoms)[*nxatoms].crdsc[m] = allatoms[cta].crdsc[m];
				}
				///and then connect the kd tree'd self refs to this xatoms ref
				(*xatoms)[*nxatoms].self = &(*xatoms)[*nxatoms];
				allatoms[cta].self = &(*xatoms)[*nxatoms];
				(*nxatoms)++;
			}
			allatoms[i].pairs[k] = &(allatoms[cta]);
			(*atoms)[ii].pairs[k] = allatoms[cta].self;

			k++;
			skip:
			kd_res_next(res);
		}
		kd_res_free(res);

		///--- Set Trips ------------------------------------------------------
#define ai (*atoms)[ii]
		///ntrips is an an array whose jth index gives the number of trips 
		///coming from pair ij.  
		ai.ntrips = calloc(ai.npairs, sizeof(unsigned int));
		ai.trips = malloc(ai.npairs*sizeof(atom**));

		for(j = 0u; j < ai.npairs; ++j){
#define aj (*atoms)[ii].pairs[j]
			ij[0] = aj->crdsc[0] - ai.crdsc[0];
			ij[1] = aj->crdsc[1] - ai.crdsc[1];
			ij[2] = aj->crdsc[2] - ai.crdsc[2];
			nij = norm_v(ij);

			///Here, we begin filling trips of j
			ai.trips[j] = malloc(ai.npairs*sizeof(atom*));
			for(k = 0u; k < ai.npairs; ++k){
#define ak (*atoms)[ii].pairs[k]
				if(j == k) continue;
				ik[0] = ak->crdsc[0] - ai.crdsc[0];
				ik[1] = ak->crdsc[1] - ai.crdsc[1];
				ik[2] = ak->crdsc[2] - ai.crdsc[2];

				///A triplet of j is formed for j-i, k-i < rcut and the angle
				///jik < acut for j != k
				///For an angle theta, 0 <= theta <= pi, we can check that
				///theta < theta_cut by checking cos(theta) > cos(theta_cut)
				///(not a typo! cos is a decreasing function in these bounds)
				///cos(angle jik) is ( (j-i)@(k-i) ) / ( |(j-i)||(k-i)| )
				if(cosacut < dot_vv(ij,ik)/(nij*norm_v(ik))){
					ai.trips[j][ai.ntrips[j]] = ak; ////ak is already a pointer
					ai.ntrips[j]++;
				}
#undef ak
			}

			///We've allocated more memory than needed - get some back		
			ai.trips[j] = realloc(ai.trips[j], ai.ntrips[j]*sizeof(atom*));
#undef aj
		}
#undef ai
	}

	
	//--- Reduce, set imgs ----------------------------------------------------
	for(i = nimgstot-1u, ii = 0u; i < nallatoms; i += nimgstot, ++ii){
		(*atoms)[ii].nimgs = 0u;
		(*atoms)[ii].imgs = malloc((nimgstot - 1u)*sizeof(atom*));

		///Set imgs to keep by checking against the "throw me away" flag
		for(j = 0u; j < allatoms[i].nimgs; ++j){
			if(allatoms[i].imgs[j]->nimgs == IS_IMAGE) continue;
			//If we're here, we want to actually keep this one
			(*atoms)[ii].imgs[(*atoms)[ii].nimgs] = allatoms[i].imgs[j]->self;
			(*atoms)[ii].nimgs++;
		}
		///We've allocated too much memory, get some back
		(*atoms)[ii].imgs = realloc((*atoms)[ii].imgs, k*sizeof(atom*));
	}
	

	//Finish up
	*xatoms = realloc(*xatoms, (*nxatoms)*sizeof(atom));
	for(i = nimgstot-1u; i < nallatoms; i += nimgstot){
		free(allatoms[i].imgs);
		free(allatoms[i].pairs);
	}
	free(allatoms);
	kd_free(kdnode);
	
	return 0;
}
#undef IS_IMAGE 
#undef IS_IMPORTANT_IMAGE 


//Completly frees the atoms arrays that were set by BOTH latticeparse() and 
//setnbrs_a().  Use at end of program, not for resetting neighbor counts
void free_a(const unsigned int natoms, atom*restrict*restrict atoms,
			atom*restrict*restrict xatoms){
	for(unsigned int i = 0u; i < natoms; ++i){
		free((*atoms)[i].spec);
		free((*atoms)[i].imgs);
		free((*atoms)[i].pairs);
		
		for(unsigned int j = 0u; j < (*atoms)[i].npairs; ++j){
			free((*atoms)[i].trips[j]);
		}
		free((*atoms)[i].ntrips);
		free((*atoms)[i].trips);
	}
	free(*atoms);
	free(*xatoms); ///no member of xatoms has any dynamic memory allocated
}


#ifdef restrict
#undef restrict
#endif
#ifdef inline
#undef inline
#endif