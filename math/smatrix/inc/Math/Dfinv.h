// @(#)root/smatrix:$Id$
// Authors: T. Glebe, L. Moneta    2005

#ifndef ROOT_Math_Dfinv
#define ROOT_Math_Dfinv
// ********************************************************************
//
// source:
//
// type:      source code
//
// created:   03. Apr 2001
//
// author:    Thorsten Glebe
//            HERA-B Collaboration
//            Max-Planck-Institut fuer Kernphysik
//            Saupfercheckweg 1
//            69117 Heidelberg
//            Germany
//            E-mail: T.Glebe@mpi-hd.mpg.de
//
// Description: Matrix inversion
//              Code was taken from CERNLIB::kernlib dfinv function, translated
//              from FORTRAN to C++ and optimized.
//
// changes:
// 03 Apr 2001 (TG) creation
//
// ********************************************************************


namespace ROOT {

  namespace Math {




/** Dfinv.
    Function to compute the inverse of a square matrix ($A^{-1}$) of
    dimension $idim$ and order $n$. The routine Dfactir must be called
    before Dfinv!

    @author T. Glebe
*/
template <class Matrix, unsigned int n, unsigned int idim>
bool Dfinv(Matrix& rhs, unsigned int* ir) {
#ifdef XXX
  if (idim < n || n <= 0 || n==1) {
    return false;
  }
#endif

  typename Matrix::value_type* a = rhs.Array();

  /* Local variables */
  unsigned int nxch, i, j, k, m, ij;
  unsigned int im2, nm1, nmi;
  typename Matrix::value_type s31, s34, ti;

  /* Parameter adjustments */
  a -= idim + 1;
  --ir;

  /* Function Body */

  /* finv.inc */

  a[idim + 2] = -a[(idim << 1) + 2] * (a[idim + 1] * a[idim + 2]);
  a[(idim << 1) + 1] = -a[(idim << 1) + 1];

   if (n != 2) {
      for (i = 3; i <= n; ++i) {
         const unsigned int ii   = i * idim;
         const unsigned int iii  = i + ii;
         const unsigned int imi  = ii - idim;
         const unsigned int iimi = i + imi;
         im2 = i - 2;
         for (j = 1; j <= im2; ++j) {
            const unsigned int ji  = j * idim;
            const unsigned int jii = j + ii;
            s31 = 0.;
            for (k = j; k <= im2; ++k) {
               s31 += a[k + ji] * a[i + k * idim];
               a[jii] += a[j + (k + 1) * idim] * a[k + 1 + ii];
            } // for k
            a[i + ji] = -a[iii] * (a[i - 1 + ji] * a[iimi] + s31);
            a[jii] *= -1;
         } // for j
         a[iimi] = -a[iii] * (a[i - 1 + imi] * a[iimi]);
         a[i - 1 + ii] *= -1;
      } // for i
   } // if n!=2

   nm1 = n - 1;
   for (i = 1; i <= nm1; ++i) {
      const unsigned int ii = i * idim;
      nmi = n - i;
      for (j = 1; j <= i; ++j) {
         const unsigned int ji  = j * idim;
         const unsigned int iji = i + ji;
         for (k = 1; k <= nmi; ++k) {
            a[iji] += a[i + k + ji] * a[i + (i + k) * idim];
         } // for k
      } // for j

      for (j = 1; j <= nmi; ++j) {
         const unsigned int ji = j * idim;
         s34 = 0.;
         for (k = j; k <= nmi; ++k) {
            s34 += a[i + k + ii + ji] * a[i + (i + k) * idim];
         } // for k
         a[i + ii + ji] = s34;
      } // for j
   } // for i

   nxch = ir[n];
   if (nxch == 0) {
      return true;
   }

   for (m = 1; m <= nxch; ++m) {
      k = nxch - m + 1;
      ij = ir[k];
      i = ij / 4096;
      j = ij % 4096;
      const unsigned int ii = i * idim;
      const unsigned int ji = j * idim;
      for (k = 1; k <= n; ++k) {
         ti = a[k + ii];
         a[k + ii] = a[k + ji];
         a[k + ji] = ti;
      } // for k
   } // for m

   return true;
} // Dfinv


  }  // namespace Math

}  // namespace ROOT



#endif  /* ROOT_Math_Dfinv */
