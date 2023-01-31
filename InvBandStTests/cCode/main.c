#include <stdlib.h>
#include <stdio.h>

#include "io.h"
#include "atomstruct.h"


int main(int argc, char** argv){
	unsigned int nspec; char** specmap; unsigned int* countmap;
	double A[3][3]; 
	unsigned int natoms; atom* atoms; unsigned int nextras; atom* extras;
	int readlatstat = latticeparse(LATTICE_INFILE, A, &nspec, &specmap, &countmap, &natoms, &atoms); 

	for(unsigned int i = 0u; i < natoms; ++i) carcrds_a(&(atoms[i]), A);
	int nbrstat = setnbrs_a(natoms, &atoms, &nextras, &extras, A, 3.5, 80.0*0.017453293, 1.0e-4);



	for(unsigned int i = 0u; i < nspec; ++i) free(specmap[i]); free(specmap);
	free(countmap);
	free_a(natoms, &atoms, &extras);

	return 0;
}