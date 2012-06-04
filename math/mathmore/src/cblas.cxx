// import just a cblas symbol in mathmore
#include "gsl/gsl_cblas.h"

namespace ROOT { 

   namespace Math { 

      namespace Blas { 

         // multiplication C = A * B where (n,m) is the size of C, A is size (n,k) and B is size (k,m)
         void  AMultB(int n, int m, int k, const double * A, const double * B, double *C) { 
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, k, 1.0, A, k, B, m, 0.0, C, m);
         }
         // multiplication C = AT * B where (n,m) is the size of C, A is size (k,n) and B is size (k,m)
         void  ATMultB(int n, int m, int k, const double * A, const double * B, double *C) { 
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, m, k, 1.0, A, n, B, m, 0.0, C, m);
         }

      }
   }
}
