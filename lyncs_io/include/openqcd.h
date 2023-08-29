#pragma once

#include <assert.h>
#include <stdlib.h>
#include <complex.h>

namespace lyncs_io {
  /*
   *  @brief From openqcd to continous format or the reverse
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


  void to_openqcd(void* in, void* out, int ndims, int* dims, bool swap=0, int ncol=3) {
    
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
	  memcpy((char*) out+((v/2)*ndims+mu)*2*size, (char*) in+(v*ndims+mu)*size, size);
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
	  memcpy((char*) out+(((vp/2)*ndims+mu)*2+1)*size, (char*) in+(v*ndims+mu)*size, size);
	}
      }
    }
  }
    int idx(int* pos, int mu, int* dims, int ndims, int ncol);
    void matmul(std::complex<double>* M, std::complex<double>* A, std::complex<double>* B, int ncol);

    double plaquette(std::complex<double>* U, int ndims, int* dims, int ncol) {
        /*
         * Calculate the average value of the real part of the trace of
         * the plaquette.
         * @param[U] Lattice data.
         * @param[ndims] Number of spacetime dimensions.
         * @param[dims] List of length ndims containing extent of dimensions.
         * @param[ncol] Number of colors.
         * @returns (double) average value of the real trace of the plaquette.
         * */
        
        int volume = 1;

        for(int i=0; i<ndims; i++){
            volume *= dims[i];
        }

        double ReTrP = 0.0;
        
        int x_pos[ndims];    //x position vector

        for(int i=0; i < ndims; i++){x_pos[i] = 0;} //Start at origin

        // Iterate over the lattice
        for(int n=0; n < (int)volume; n++)
        { 
            for(int mu = 0; mu < ndims; mu++)
            {
                int xmu[ndims]; // Step in mu direction
                memcpy(xmu, x_pos, ndims*sizeof(int));
                xmu[mu] += 1;
                xmu[mu] %= dims[mu];
                
                for(int nu=mu+1; nu < ndims; nu++){
                    
                    std::complex<double> U_mu_U_nu_xmu[ncol*ncol];
                    std::complex<double> U_nu_U_mu_xnu[ncol*ncol];

                    int xnu[ndims]; //Step in nu direction
                    memcpy(xnu, x_pos, ndims*sizeof(int));
                    xnu[nu] += 1;
                    xnu[nu] %= dims[nu];               

                    // Calculate U_mu(x)*U_nu(xmu)
                    std::complex<double>* U_mu_idx = U+idx(x_pos, mu, dims, ndims, ncol);
                    std::complex<double>* U_nu_xmu_idx = U+idx(xmu, nu, dims, ndims, ncol);
                    
                    matmul(U_mu_U_nu_xmu, U_mu_idx, U_nu_xmu_idx, ncol);

                    // Calculate U_nu(x)*U_mu(xnu)                    
                    std::complex<double>* U_nu_idx = U+idx(x_pos, nu, dims, ndims, ncol);
                    std::complex<double>* U_mu_xnu_idx = U+idx(xnu, mu, dims, ndims, ncol); 

                    matmul(U_nu_U_mu_xnu, U_nu_idx, U_mu_xnu_idx, ncol);

                    // Calculate plaquette
                    for(int c1=0; c1<ncol; c1++){
                        for(int c2=0; c2<ncol; c2++){
                            {
                                ReTrP += std::real(*(U_mu_U_nu_xmu + c1*ncol + c2) * 
                                        std::conj(*(U_nu_U_mu_xnu + c1*ncol + c2)));
                            }
                        }
                    }
                }
            }

            // Move to next lattice site

            int site_set = 0;
            int dim = 0;

            while(site_set == 0 && n < volume){
                x_pos[dim] += 1;
                if(x_pos[dim] == dims[dim]){
                    x_pos[dim] = 0;
                    dim += 1;
                }else{
                    dim = 0;
                    site_set = 1;
                }
            }
        }
        
        // Calculate number of plaquette planes
        int nplanes = ndims*(ndims-1)/2;

        // Calculate average value
        ReTrP /= (volume * nplanes);

        //memcpy((double*) out, (double*) &ReTrP, sizeof(double));
        return (double) ReTrP;

    }

    // plaquette func here

    int idx(int* pos, int mu, int* dims, int ndims, int ncol){
        // Calculate the row-major address in memory of an ncol x ncol
        // matrix at position pos and direction mu
        //
        // Assumes the shape of the data is dims * ndims * ncol * ncol
        
        int idx = 0;
        
        for(int k=0; k < ndims; k++){
            int prod_dim = 1;

            for(int l=k+1; l < ndims; l++){
                prod_dim *= dims[l];
            }
            prod_dim *= (ndims*ncol*ncol);
            idx += prod_dim*pos[k];
        }
        return idx + mu*ncol*ncol;
    }
    void matmul(std::complex<double>* M, std::complex<double>* A, std::complex<double>* B, int ncol){
        // Calculate the product of two matrices A and B, and store in M.

        std::complex<double> sum = 0;

        for(int i=0; i < ncol; i++){
            for(int j=0; j < ncol; j++){
                sum = 0;
                for(int k=0; k < ncol; k++){
                    sum += *(A+i*ncol+k) * *(B+k*ncol+j);
                }
                *(M+i*ncol + j) = sum;
                
            }
        }
    }
}
