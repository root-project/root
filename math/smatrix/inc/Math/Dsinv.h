// @(#)root/smatrix:$Id$
// Authors: T. Glebe, L. Moneta    2005

#ifndef  ROOT_Math_Dsinv
#define  ROOT_Math_Dsinv
// ********************************************************************
//
// source:
//
// type:      source code
//
// created:   22. Mar 2001
//
// author:    Thorsten Glebe
//            HERA-B Collaboration
//            Max-Planck-Institut fuer Kernphysik
//            Saupfercheckweg 1
//            69117 Heidelberg
//            Germany
//            E-mail: T.Glebe@mpi-hd.mpg.de
//
// Description: Inversion of a symmetric, positive definite matrix.
//              Code was taken from CERNLIB::kernlib dsinv function, translated
//              from FORTRAN to C++ and optimized.
//
// changes:
// 22 Mar 2001 (TG) creation
//
// ********************************************************************

#include "Math/SMatrixDfwd.h"

namespace ROOT {

  namespace Math {

/** Dsinv.
    Compute inverse of a symmetric, positive definite matrix of dimension
    \f$idim\f$ and order \f$n\f$.

    @author T. Glebe
*/
template <class T, int n, int idim>
class SInverter
{

public:
  template <class MatrixRep>
  inline static bool Dsinv(MatrixRep& rhs) {

    /* Local variables */
    int i, j, k, l;
    T s31, s32;
    int jm1, jp1;

    /* Parameter adjustments */
    const int arrayOffset = -1*(idim + 1);


    /* Function Body */
    if (idim < n || n <= 1) {
      return false;
    }

    /* sfact.inc */
    for (j = 1; j <= n; ++j) {
      const int ja  = j * idim;
      const int jj  = j + ja;
      const int ja1 = ja + idim;


      if (rhs[jj + arrayOffset] <= 0.) { return false; }
      rhs[jj + arrayOffset] = 1. / rhs[jj + arrayOffset];
      if (j == n) { break; }

      for (l = j + 1; l <= n; ++l) {
        rhs[j + (l * idim) + arrayOffset] = rhs[jj + arrayOffset] * rhs[l + ja + arrayOffset];
        const int lj = l + ja1;
        for (i = 1; i <= j; ++i) {
          rhs[lj + arrayOffset] -= rhs[l + (i * idim)  + arrayOffset] * rhs[i + ja1 + arrayOffset];
        }
      }
    }

    /* sfinv.inc */
    // idim << 1 is equal to idim * 2
    // compiler will compute the arguments!
    rhs[((idim << 1) + 1) + arrayOffset] = -rhs[((idim << 1) + 1) + arrayOffset];
    rhs[idim + 2 + arrayOffset] = rhs[((idim << 1)) + 1 + arrayOffset] * rhs[((idim << 1)) + 2 + arrayOffset];

    if(n > 2) {

      for (j = 3; j <= n; ++j) {
        const int jm2 = j - 2;
        const int ja = j * idim;
        const int jj = j + ja;
        const int j1 = j - 1 + ja;

        for (k = 1; k <= jm2; ++k) {
          s31 = rhs[k + ja + arrayOffset];

          for (i = k; i <= jm2; ++i) {
            s31 += rhs[k + ((i + 1) * idim) + arrayOffset] * rhs[i + 1 + ja + arrayOffset];
          } // for i
          rhs[k + ja + arrayOffset] = -s31;
          rhs[j + (k * idim) + arrayOffset] = -s31 * rhs[jj + arrayOffset];
        } // for k
        rhs[j1 + arrayOffset] *= -1;
        //      rhs[j1] = -rhs[j1];
        rhs[jj - idim + arrayOffset] = rhs[j1 + arrayOffset] * rhs[jj + arrayOffset];
      } // for j
    } // if (n>2)

    j = 1;
    do {
      const int jad = j * idim;
      const int jj = j + jad;

      jp1 = j + 1;
      for (i = jp1; i <= n; ++i) {
        rhs[jj + arrayOffset] += rhs[j + (i * idim) + arrayOffset] * rhs[i + jad + arrayOffset];
      } // for i

      jm1 = j;
      j = jp1;
      const int ja = j * idim;

      for (k = 1; k <= jm1; ++k) {
        s32 = 0.;
        for (i = j; i <= n; ++i) {
          s32 += rhs[k + (i * idim) + arrayOffset] * rhs[i + ja + arrayOffset];
        } // for i
        //rhs[k + ja + arrayOffset] = rhs[j + (k * idim) + arrayOffset] = s32;
        rhs[k + ja + arrayOffset] = s32;
      } // for k
    } while(j < n);

    return true;
  }


    // for symmetric matrices

  static bool Dsinv(MatRepSym<T,n> & rhs) {
    // not very efficient but need to re-do Dsinv for new storage of
    // symmetric matrices
    MatRepStd<T,n,n> tmp;
    for (int i = 0; i< n*n; ++i)
      tmp[i] = rhs[i];
    // call dsinv
    if (! SInverter<T,n,n>::Dsinv(tmp) ) return false;
    //if (! Inverter<n>::Dinv(tmp) ) return false;
    // recopy the data
    for (int i = 0; i< n*n; ++i)
      rhs[i] = tmp[i];

    return true;

  }

}; // end of Dsinv



  }  // namespace Math

}  // namespace ROOT


#endif  /* ROOT_Math_Dsinv */
