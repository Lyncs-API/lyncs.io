#pragma once

#include <assert.h>

namespace lyncs_io {
  /*
   *  @brief From openqcd to continous format
   *  @param[out] Same size as input but with sites sorted first even and then odd
   *  @param[in] A field of size prod(dims)*ndims*ncol*2*sizeof(double)
   *  @param[ndims] Number of lattice dimensions
   *  @param[dims] Lattice size
   *  @param[swap] Swaps even with odd in the output/input
   *  @param[ncol] Number of colors
   */
  void from_openqcd(void* out, void* in, int ndims, int* dims, bool swap=0, int ncol=3) {
    
    int size = ncol*ncol*2*sizeof(double);
      
    size_t faces[ndims];
    size_t volume = 1;
    for (int i=ndims-1; i>=0; i--) {
      faces[i] = volume;
      volume *= dims[i];
    }

    for(size_t v=0; v<volume; v++) {
	
      size_t tmp = v;
      bool isodd = swap;
      for (int i=ndims-1; i>=0; i--) {
	isodd ^= tmp % dims[i] % 2;
	tmp /= dims[i];
      }
	
      if (isodd) {
	for(int mu=0; mu<ndims; mu++) {
	  memcpy((char*) out+(v*ndims+mu)*size, (char*) in+((v/2)*ndims+mu)*2*size, size);
	}
      }
      else {
	for(int mu=0; mu<ndims; mu++) {
	  int l = (v/faces[mu])%dims[mu];
	  size_t vp = v;
	  if(l+1 == dims[mu]) 
	    vp -= l*faces[mu];
	  else
	    vp += faces[mu];
	  memcpy((char*) out+(v*ndims+mu)*size, (char*) in+(((vp/2)*ndims+mu)*2+1)*size, size);
	}	  
      }
    }
  }
}
