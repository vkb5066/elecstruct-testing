#ifndef IO_H
#define IO_H
/*
* Header file for all input/output functions
*/
#include "atomstruct.h"

#define LATTICE_INFILE "lattice.esc"
#define SPEC_MAX_LEN 4u

#ifdef __unix__
#define restrict restrict 
#define inline static __attribute__((always_inline))
#endif
#ifdef _WIN32
#define restrict __restrict 
#define inline static __forceinline 
#endif


typedef struct cla cla;
struct cla{
	unsigned int runmode;
};

//Parses the command line and stores all arguments
//Returns the number of successful updates
int claparse(cla*restrict args, 
			 const int argc, const char*restrict*restrict argv);
//Reads lattice from input file, updates atoms, basis, and element mapping
//returns 0 on success, 1 if file was not found
int latticeparse(const char*restrict filename,
				 double A[restrict 3][restrict 3], 
				 unsigned int*restrict nspecs, 
				 char*restrict*restrict*restrict specmap, 
				 unsigned int*restrict*restrict countmap,
				 unsigned int*restrict natoms, atom*restrict*restrict atoms);


#ifdef restrict
#undef restrict
#endif
#ifdef inline
#undef inline
#endif

#endif