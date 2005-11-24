// @(#)root/smatrix:$Name:  $:$Id: Dsfact.hv 1.0 2005/11/24 12:00:00 moneta Exp $
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


namespace ROOT { 

  namespace Math { 




/** Dsfact.
    Compute determinant of a symmetric, positive definite matrix of dimension
    $idim$ and order $n$.
    
    @author T. Glebe
*/
template <class Matrix, unsigned int n, unsigned int idim>
bool Dsfact(Matrix& rhs, typename Matrix::value_type& det) {

#ifdef XXX
  /* Function Body */
  if (idim < n || n <= 0) {
    return false;
  }
#endif

  typename Matrix::value_type* a = rhs.Array();

#ifdef XXX
  const typename Matrix::value_type* A = rhs.Array();
  typename Matrix::value_type array[Matrix::size];
  typename Matrix::value_type* a = array;

  // copy contents of matrix to working place
  for(unsigned int i=0; i<Matrix::size; ++i) {
    array[i] = A[i];
  }
#endif

  /* Local variables */
  static unsigned int i, j, l;

  /* Parameter adjustments */
  a -= idim + 1;

  /* sfactd.inc */
  det = 1.;
  for (j = 1; j <= n; ++j) {
    const unsigned int ji = j * idim;
    const unsigned int jj = j + ji;

    if (a[jj] <= 0.) {
      det = 0.;
      return false;
    }

    const unsigned int jp1 = j + 1;
    const unsigned int jpi = jp1 * idim;

    det *= a[jj];
    a[jj] = 1. / a[jj];

    for (l = jp1; l <= n; ++l) {
      a[j + l * idim] = a[jj] * a[l + ji];

      const unsigned int lj = l + jpi;

      for (i = 1; i <= j; ++i) {
	a[lj] -= a[l + i * idim] * a[i + jpi];
      } // for i
    } // for l
  } // for j

  return true;
} // end of Dsfact


  }  // namespace Math

}  // namespace ROOT

#endif  /* ROOT_Math_Dsfact */

