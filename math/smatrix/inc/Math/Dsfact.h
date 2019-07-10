// @(#)root/smatrix:$Id$
// Authors: T. Glebe, L. Moneta    2005

#ifndef ROOT_Math_Dsfact
#define ROOT_Math_Dsfact
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
// Description: Determinant of a symmetric, positive definite matrix.
//              Code was taken from CERNLIB::kernlib dsfact function, translated
//              from FORTRAN to C++ and optimized.
//
// changes:
// 22 Mar 2001 (TG) creation
// 18 Apr 2001 (TG) removed internal copying of array.
//
// ********************************************************************

#include "Math/MatrixRepresentationsStatic.h"

namespace ROOT {

  namespace Math {




/** Dsfact.
    Compute determinant of a symmetric, positive definite matrix of dimension
    \f$idim\f$ and order \f$n\f$.

    @author T. Glebe
*/

template <unsigned int n, unsigned int idim =n>
class SDeterminant {

public:
template <class T>
static bool Dsfact(MatRepStd<T,n,idim>& rhs, T& det) {

#ifdef XXX
  /* Function Body */
  if (idim < n || n <= 0) {
    return false;
  }
#endif

#ifdef OLD_IMPL
  typename MatrixRep::value_type* a = rhs.Array();
#endif

#ifdef XXX
  const typename MatrixRep::value_type* A = rhs.Array();
  typename MatrixRep::value_type array[MatrixRep::kSize];
  typename MatrixRep::value_type* a = array;

  // copy contents of matrix to working place
  for(unsigned int i=0; i<MatrixRep::kSize; ++i) {
    array[i] = A[i];
  }
#endif

  /* Local variables */
  unsigned int i, j, l;

  /* Parameter adjustments */
  //  a -= idim + 1;
  const int arrayOffset = -(idim+1);
  /* sfactd.inc */
  det = 1.;
  for (j = 1; j <= n; ++j) {
    const unsigned int ji = j * idim;
    const unsigned int jj = j + ji;

    if (rhs[jj + arrayOffset] <= 0.) {
      det = 0.;
      return false;
    }

    const unsigned int jp1 = j + 1;
    const unsigned int jpi = jp1 * idim;

    det *= rhs[jj + arrayOffset];
    rhs[jj + arrayOffset] = 1. / rhs[jj + arrayOffset];

    for (l = jp1; l <= n; ++l) {
      rhs[j + l * idim + arrayOffset] = rhs[jj + arrayOffset] * rhs[l + ji + arrayOffset];

      const unsigned int lj = l + jpi;

      for (i = 1; i <= j; ++i) {
         rhs[lj + arrayOffset] -= rhs[l + i * idim + arrayOffset] * rhs[i + jpi + arrayOffset];
      } // for i
    } // for l
  } // for j

  return true;
} // end of Dsfact


   // t.b.d re-implement methods for symmetric
  // symmetric function (copy in a general  one)
  template <class T>
  static bool Dsfact(MatRepSym<T,n> & rhs,  T & det) {
    // not very efficient but need to re-do Dsinv for new storage of
    // symmetric matrices
    MatRepStd<T,n> tmp;
    for (unsigned int i = 0; i< n*n; ++i)
      tmp[i] = rhs[i];
    if (!  SDeterminant<n>::Dsfact(tmp,det) ) return false;
//     // recopy the data
//     for (int i = 0; i< idim*n; ++i)
//       rhs[i] = tmp[i];

    return true;
  }


};  // end of class Sdeterminant

  }  // namespace Math

}  // namespace ROOT

#endif  /* ROOT_Math_Dsfact */

