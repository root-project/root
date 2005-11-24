// @(#)root/smatrix:$Name:  $:$Id: Dsinv.hv 1.0 2005/11/24 12:00:00 moneta Exp $
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


namespace ROOT { 

  namespace Math { 

/** Dsinv.
    Compute inverse of a symmetric, positive definite matrix of dimension
    $idim$ and order $n$.

    @author T. Glebe
*/
template <class T, int n, int idim>
bool Dsinv(T* a) {

  /* Local variables */
  static int i, j, k, l;
  static T s31, s32;
  static int jm1, jp1;

  /* Parameter adjustments */
  a -= idim + 1;

  /* Function Body */
  if (idim < n || n <= 1) {
    return false;
  }

  /* sfact.inc */
  for (j = 1; j <= n; ++j) {
    const int ja  = j * idim;
    const int jj  = j + ja;
    const int ja1 = ja + idim;

    if (a[jj] <= 0.) { return false; }
    a[jj] = 1. / a[jj];
    if (j == n) { break; }

    for (l = j + 1; l <= n; ++l) {
      a[j + l * idim] = a[jj] * a[l + ja];
      const int lj = l + ja1;
      for (i = 1; i <= j; ++i) {
	a[lj] -= a[l + i * idim] * a[i + ja1];
      }
    }
  }

  /* sfinv.inc */
  // idim << 1 is equal to idim * 2
  // compiler will compute the arguments!
  a[(idim << 1) + 1] = -a[(idim << 1) + 1];
  a[idim + 2] = a[(idim << 1) + 1] * a[(idim << 1) + 2];

  if(n > 2) {

    for (j = 3; j <= n; ++j) {
      const int jm2 = j - 2;
      const int ja = j * idim;
      const int jj = j + ja;
      const int j1 = j - 1 + ja;

      for (k = 1; k <= jm2; ++k) {
	s31 = a[k + ja];

	for (i = k; i <= jm2; ++i) {
	  s31 += a[k + (i + 1) * idim] * a[i + 1 + ja];
	} // for i
	a[k + ja] = -s31;
	a[j + k * idim] = -s31 * a[jj];
      } // for k
      a[j1] *= -1;
      //      a[j1] = -a[j1];
      a[jj - idim] = a[j1] * a[jj];
    } // for j
  } // if (n>2)

  j = 1;
  do {
    const int jad = j * idim;
    const int jj = j + jad;

    jp1 = j + 1;
    for (i = jp1; i <= n; ++i) {
      a[jj] += a[j + i * idim] * a[i + jad];
    } // for i

    jm1 = j;
    j = jp1;
    const int ja = j * idim;

    for (k = 1; k <= jm1; ++k) {
      s32 = 0.;
      for (i = j; i <= n; ++i) {
	s32 += a[k + i * idim] * a[i + ja];
      } // for i
      a[k + ja] = a[j + k * idim] = s32;
    } // for k
  } while(j < n);

  return true;
} // end of Dsinv


  }  // namespace Math

}  // namespace ROOT
          

#endif  /* ROOT_Math_Dsinv */
