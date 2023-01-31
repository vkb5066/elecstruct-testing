#pragma warning(disable:4996)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io.h"

#ifdef __unix__
#define restrict restrict 
#define inline static __attribute__((always_inline))
#endif
#ifdef _WIN32
#define restrict __restrict 
#define inline static __forceinline 
#endif

#define LINESIZE_MAX 128u
#define N_TOKENS_MAX 32u
#define DELIM " ,\t\n\r"
#define BASE 10

//Reads lattice from input file, updates atoms, basis, and element mapping
//returns 0 on success, 1 if file was not found
int latticeparse(const char*restrict filename,
				 double A[restrict 3][restrict 3], 
				 unsigned int*restrict nspecs, 
				 char*restrict*restrict*restrict specmap, 
				 unsigned int*restrict*restrict countmap,
				 unsigned int*restrict natoms, atom*restrict*restrict atoms){
	FILE* infile;
	infile = fopen(filename, "r");
	if(!infile) return 1;

	char line[LINESIZE_MAX];
	char* next;


	//Unit / supercell definition
	for(unsigned int i = 0u; i < 3u; ++i){
		fgets(line, LINESIZE_MAX, infile);
		A[i][0] = strtod(line, &next);
		A[i][1] = strtod(next, &next);
		A[i][2] = strtod(next, NULL);
	}

	//Species types
	*nspecs = 0u;
	*specmap = malloc(N_TOKENS_MAX*sizeof(char*));
	for(unsigned int i = 0u; i < N_TOKENS_MAX; ++i){
		(*specmap)[i] = malloc(SPEC_MAX_LEN*sizeof(char));
	}
	fgets(line, LINESIZE_MAX, infile);
	next = strtok(line, DELIM);
	while(next){
		for(unsigned int i = 0u; i < SPEC_MAX_LEN; ++i){
			(*specmap)[(*nspecs)][i] = next[i]; 
		}
		(*nspecs)++;
		next = strtok(NULL, DELIM);
	}
	for(unsigned int i = (*nspecs); i < N_TOKENS_MAX; ++i){
		free((*specmap)[i]);
	}
	*specmap = realloc(*specmap, (*nspecs)*sizeof(char*));

	//Species counts
	*countmap = calloc(*nspecs, sizeof(unsigned int));
	fgets(line, LINESIZE_MAX, infile);
	(*countmap)[0] = (unsigned int)strtoul(line, &next, BASE);
	*natoms = (*countmap)[0];
	for(unsigned int i = 1u; i < (*nspecs); ++i){
		(*countmap)[i] = (unsigned int)strtoul(next, &next, BASE);
		(*natoms) += (*countmap)[i];
	}

	//Atoms
	*atoms = malloc((*natoms)*sizeof(atom));
	for(unsigned int i = 0u, count = 0u; i < *nspecs; ++i){
		for(unsigned int j = 0u; j < (*countmap)[i]; ++j, ++count){
			fgets(line, LINESIZE_MAX, infile);

			///Prep this atom
			(*atoms)[count].spec = malloc(sizeof(unsigned int));
			*(*atoms)[count].spec = i;
			(*atoms)[count].crdsf[0] = strtod(line, &next);
			(*atoms)[count].crdsf[1] = strtod(next, &next);
			(*atoms)[count].crdsf[2] = strtod(next, NULL);
			(*atoms)[count].self = &(*atoms)[count];
		}
	}

	fclose(infile);
	return 0;
}


#ifdef restrict
#undef restrict
#endif
#ifdef inline
#undef inline
#endif
#undef LINESIZE_MAX
#undef N_TOKENS_MAX
#undef DELIM
#undef BASE
