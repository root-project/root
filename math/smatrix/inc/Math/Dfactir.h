// @(#)root/smatrix:$Id$
// Authors: T. Glebe, L. Moneta    2005

#ifndef ROOT_Math_Dfactir
#define ROOT_Math_Dfactir
// ********************************************************************
//
// source:
//
// type:      source code
//
// created:   02. Apr 2001
//
// author:    Thorsten Glebe
//            HERA-B Collaboration
//            Max-Planck-Institut fuer Kernphysik
//            Saupfercheckweg 1
//            69117 Heidelberg
//            Germany
//            E-mail: T.Glebe@mpi-hd.mpg.de
//
// Description: Determinant of a square matrix, needed for Dfinv()
//              Code was taken from CERNLIB::kernlib dfact function, translated
//              from FORTRAN to C++ and optimized.
//
// changes:
// 02 Apr 2001 (TG) creation
//
// ********************************************************************

#include <cmath>

namespace ROOT {

  namespace Math {


/** Dfactir.
    Function to compute the determinant from a square matrix, Det(A) of
    dimension idim and order n. A working area ir is returned which is
    needed by the Dfinv routine.

    @author T. Glebe
*/
template <class Matrix, unsigned int n, unsigned int idim>
bool Dfactir(Matrix& rhs, typename Matrix::value_type& det, unsigned int* ir)
  // int *n, float *a, int *idim, int *ir, int *ifail, float *det, int *jfail)
{

#ifdef XXX
  if (idim < n || n <= 0) {
    return false;
  }
#endif


  /* Initialized data */
   typename Matrix::value_type* a = rhs.Array();

   /* Local variables */
   unsigned int nxch, i, j, k, l;
   typename Matrix::value_type p, q, tf;

   /* Parameter adjustments */
   a -= idim + 1;
   --ir;

   /* Function Body */

   // fact.inc
   nxch = 0;
   det = 1.;
   for (j = 1; j <= n; ++j) {
      const unsigned int ji = j * idim;
      const unsigned int jj = j + ji;

      k = j;
      p = std::abs(a[jj]);

      if (j != n) {
         for (i = j + 1; i <= n; ++i) {
            q = std::abs(a[i + ji]);
            if (q > p) {
               k = i;
               p = q;
            }
         } // for i

         if (k != j) {
            for (l = 1; l <= n; ++l) {
               const unsigned int li = l*idim;
               const unsigned int jli = j + li;
               const unsigned int kli = k + li;
               tf = a[jli];
               a[jli] = a[kli];
               a[kli] = tf;
            } // for l
            ++nxch;
            ir[nxch] = (j << 12) + k;
         } // if k != j
      } // if j!=n

      if (p <= 0.) {
         det = 0;
         return false;
      }

      det *= a[jj];
#ifdef XXX
      t = std::abs(det);
      if (t < 1e-19 || t > 1e19) {
         det = 0;
         return false;
      }
#endif

      a[jj] = 1. / a[jj];
      if (j == n) {
         continue;
      }

      const unsigned int jm1 = j - 1;
      const unsigned int jpi = (j + 1) * idim;
      const unsigned int jjpi = j + jpi;

      for (k = j + 1; k <= n; ++k) {
         const unsigned int ki  = k * idim;
         const unsigned int jki = j + ki;
         const unsigned int kji = k + jpi;
         if (j != 1) {
            for (i = 1; i <= jm1; ++i) {
               const unsigned int ii = i * idim;
               a[jki] -= a[i + ki] * a[j + ii];
               a[kji] -= a[i + jpi] * a[k + ii];
            } // for i
         }
         a[jki] *= a[jj];
         a[kji] -= a[jjpi] * a[k + ji];
      } // for k
   } // for j

   if (nxch % 2 != 0) {
      det = -(det);
   }
   ir[n] = nxch;
   return true;
} // end of Dfact


  }  // namespace Math

}  // namespace ROOT



#endif /* ROOT_Math_Dfactir */
