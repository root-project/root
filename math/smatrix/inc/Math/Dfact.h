// @(#)root/smatrix:$Id$
// Authors: T. Glebe, L. Moneta    2005

#ifndef ROOT_Math_Dfact
#define ROOT_Math_Dfact
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
// Description: Determinant of a square matrix
//              Code was taken from CERNLIB::kernlib dfact function, translated
//              from FORTRAN to C++ and optimized.
//
// changes:
// 02 Apr 2001 (TG) creation
//
// ********************************************************************

#include <cmath>

#ifndef ROOT_Math_MatrixRepresentationsStatic
#include "Math/MatrixRepresentationsStatic.h"
#endif

namespace ROOT {

  namespace Math {



/**
    Detrminant for a general squared matrix
    Function to compute the determinant from a square matrix (\f$ \det(A)\f$) of
    dimension idim and order n.

    @author T. Glebe
*/
template <unsigned int n, unsigned int idim = n>
class Determinant {
public:

template <class T>
static bool Dfact(MatRepStd<T,n,idim>& rhs, T& det) {

#ifdef XXX
  if (idim < n || n <= 0) {
    return false;
  }
#endif


  /* Initialized data */
  //  const typename MatrixRep::value_type* A = rhs.Array();
  //typename MatrixRep::value_type* a = rhs.Array();

  /* Local variables */
  unsigned int nxch, i, j, k, l;
  //static typename MatrixRep::value_type p, q, tf;
  T p, q, tf;

  /* Parameter adjustments */
  //  a -= idim + 1;
  const int arrayOffset = - int(idim+1);
  /* Function Body */

  // fact.inc

   nxch = 0;
   det = 1.;
   for (j = 1; j <= n; ++j) {
      const unsigned int ji = j * idim;
      const unsigned int jj = j + ji;

      k = j;
      p = std::abs(rhs[jj + arrayOffset]);

      if (j != n) {
         for (i = j + 1; i <= n; ++i) {
            q = std::abs(rhs[i + ji + arrayOffset]);
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
               tf = rhs[jli + arrayOffset];
               rhs[jli + arrayOffset] = rhs[kli + arrayOffset];
               rhs[kli + arrayOffset] = tf;
            } // for l
            ++nxch;
         } // if k != j
      } // if j!=n

      if (p <= 0.) {
         det = 0;
         return false;
      }

      det *= rhs[jj + arrayOffset];
#ifdef XXX
      t = std::abs(det);
      if (t < 1e-19 || t > 1e19) {
         det = 0;
         return false;
      }
#endif
      // using 1.0f removes a warning on Windows (1.0f is still the same  as 1.0)
      rhs[jj + arrayOffset] = 1.0f / rhs[jj + arrayOffset];
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
               rhs[jki + arrayOffset] -= rhs[i + ki + arrayOffset] * rhs[j + ii + arrayOffset];
               rhs[kji + arrayOffset] -= rhs[i + jpi + arrayOffset] * rhs[k + ii + arrayOffset];
            } // for i
         }
         rhs[jki + arrayOffset] *= rhs[jj + arrayOffset];
         rhs[kji + arrayOffset] -= rhs[jjpi + arrayOffset] * rhs[k + ji + arrayOffset];
      } // for k
   } // for j

   if (nxch % 2 != 0) {
      det = -(det);
  }
  return true;
} // end of Dfact


   // t.b.d re-implement methods for symmetric
  // symmetric function (copy in a general  one)
  template <class T>
  static bool Dfact(MatRepSym<T,n> & rhs, T & det) {
    // not very efficient but need to re-do Dsinv for new storage of
    // symmetric matrices
    MatRepStd<T,n> tmp;
    for (unsigned int i = 0; i< n*n; ++i)
      tmp[i] = rhs[i];
    if (! Determinant<n>::Dfact(tmp,det) ) return false;
//     // recopy the data
//     for (int i = 0; i< idim*n; ++i)
//       rhs[i] = tmp[i];

    return true;
  }

};


  }  // namespace Math

}  // namespace ROOT



#endif /* ROOT_Math_Dfact */
