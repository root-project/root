// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

/* ddot.f -- translated by f2c (version 20010320).
   You must link the resulting object file with the libraries:
   -lf2c -lm   (in that order)
*/

namespace ROOT {

namespace Minuit2 {

double mnddot(unsigned int n, const double *dx, int incx, const double *dy, int incy)
{
   /* System generated locals */
   int i__1;
   double ret_val;

   /* Local variables */
   int i__, m;
   double dtemp;
   int ix, iy, mp1;

   /*     forms the dot product of two vectors. */
   /*     uses unrolled loops for increments equal to one. */
   /*     jack dongarra, linpack, 3/11/78. */
   /*     modified 12/3/93, array(1) declarations changed to array(*) */

   /* Parameter adjustments */
   --dy;
   --dx;

   /* Function Body */
   ret_val = 0.;
   dtemp = 0.;
   if (n <= 0) {
      return ret_val;
   }
   if (incx == 1 && incy == 1) {
      goto L20;
   }

   /*        code for unequal increments or equal increments */
   /*          not equal to 1 */

   ix = 1;
   iy = 1;
   if (incx < 0) {
      ix = (-static_cast<int>(n) + 1) * incx + 1;
   }
   if (incy < 0) {
      iy = (-static_cast<int>(n) + 1) * incy + 1;
   }
   i__1 = n;
   for (i__ = 1; i__ <= i__1; ++i__) {
      dtemp += dx[ix] * dy[iy];
      ix += incx;
      iy += incy;
      /* L10: */
   }
   ret_val = dtemp;
   return ret_val;

   /*        code for both increments equal to 1 */

   /*        clean-up loop */

L20:
   m = n % 5;
   if (m == 0) {
      goto L40;
   }
   i__1 = m;
   for (i__ = 1; i__ <= i__1; ++i__) {
      dtemp += dx[i__] * dy[i__];
      /* L30: */
   }
   if (n < 5) {
      goto L60;
   }
L40:
   mp1 = m + 1;
   i__1 = n;
   for (i__ = mp1; i__ <= i__1; i__ += 5) {
      dtemp = dtemp + dx[i__] * dy[i__] + dx[i__ + 1] * dy[i__ + 1] + dx[i__ + 2] * dy[i__ + 2] +
              dx[i__ + 3] * dy[i__ + 3] + dx[i__ + 4] * dy[i__ + 4];
      /* L50: */
   }
L60:
   ret_val = dtemp;
   return ret_val;
} /* ddot_ */

} // namespace Minuit2

} // namespace ROOT
