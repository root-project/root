/* poly/zsolve_cubic.c
 *
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007 Brian Gough
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* zsolve_cubic.c - finds the complex roots of x^3 + a x^2 + b x + c = 0 */

//#include <config.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_poly.h>

#define SWAP(a,b) do { double tmp = b ; b = a ; a = tmp ; } while(0)

int
gsl_poly_complex_solve_cubic (double a, double b, double c,
                              gsl_complex *z0, gsl_complex *z1,
                              gsl_complex *z2)
{
  double q = (a * a - 3 * b);
  double r = (2 * a * a * a - 9 * a * b + 27 * c);

  double Q = q / 9;
  double R = r / 54;

  double Q3 = Q * Q * Q;
  double R2 = R * R;

//  double CR2 = 729 * r * r;
//  double CQ3 = 2916 * q * q * q;

  if (R == 0 && Q == 0)
    {
      GSL_REAL (*z0) = -a / 3;
      GSL_IMAG (*z0) = 0;
      GSL_REAL (*z1) = -a / 3;
      GSL_IMAG (*z1) = 0;
      GSL_REAL (*z2) = -a / 3;
      GSL_IMAG (*z2) = 0;
      return 3;
    }
  else if (R2 == Q3)
    {
      /* this test is actually R2 == Q3, written in a form suitable
         for exact computation with integers */

      /* Due to finite precision some double roots may be missed, and
         will be considered to be a pair of complex roots z = x +/-
         epsilon i close to the real axis. */

      double sqrtQ = sqrt (Q);

      if (R > 0)
        {
          GSL_REAL (*z0) = -2 * sqrtQ - a / 3;
          GSL_IMAG (*z0) = 0;
          GSL_REAL (*z1) = sqrtQ - a / 3;
          GSL_IMAG (*z1) = 0;
          GSL_REAL (*z2) = sqrtQ - a / 3;
          GSL_IMAG (*z2) = 0;
        }
      else
        {
          GSL_REAL (*z0) = -sqrtQ - a / 3;
          GSL_IMAG (*z0) = 0;
          GSL_REAL (*z1) = -sqrtQ - a / 3;
          GSL_IMAG (*z1) = 0;
          GSL_REAL (*z2) = 2 * sqrtQ - a / 3;
          GSL_IMAG (*z2) = 0;
        }
      return 3;
    }
  else if (R2 < Q3)  /* equivalent to R2 < Q3 */
    {
      double sqrtQ = sqrt (Q);
      double sqrtQ3 = sqrtQ * sqrtQ * sqrtQ;
      double ctheta = R / sqrtQ3;
      double theta = 0;
      if ( ctheta <= -1.0)
         theta = M_PI;
      else if ( ctheta < 1.0)
         theta = acos (R / sqrtQ3);

      double norm = -2 * sqrtQ;
      double r0 = norm * cos (theta / 3) - a / 3;
      double r1 = norm * cos ((theta + 2.0 * M_PI) / 3) - a / 3;
      double r2 = norm * cos ((theta - 2.0 * M_PI) / 3) - a / 3;

      /* Sort r0, r1, r2 into increasing order */

      if (r0 > r1)
        SWAP (r0, r1);

      if (r1 > r2)
        {
          SWAP (r1, r2);

          if (r0 > r1)
            SWAP (r0, r1);
        }

      GSL_REAL (*z0) = r0;
      GSL_IMAG (*z0) = 0;

      GSL_REAL (*z1) = r1;
      GSL_IMAG (*z1) = 0;

      GSL_REAL (*z2) = r2;
      GSL_IMAG (*z2) = 0;

      return 3;
    }
  else
    {
      double sgnR = (R >= 0 ? 1 : -1);
      double A = -sgnR * pow (fabs (R) + sqrt (R2 - Q3), 1.0 / 3.0);
      double B = Q / A;

      if (A + B < 0)
        {
          GSL_REAL (*z0) = A + B - a / 3;
          GSL_IMAG (*z0) = 0;

          GSL_REAL (*z1) = -0.5 * (A + B) - a / 3;
          GSL_IMAG (*z1) = -(sqrt (3.0) / 2.0) * fabs(A - B);

          GSL_REAL (*z2) = -0.5 * (A + B) - a / 3;
          GSL_IMAG (*z2) = (sqrt (3.0) / 2.0) * fabs(A - B);
        }
      else
        {
          GSL_REAL (*z0) = -0.5 * (A + B) - a / 3;
          GSL_IMAG (*z0) = -(sqrt (3.0) / 2.0) * fabs(A - B);

          GSL_REAL (*z1) = -0.5 * (A + B) - a / 3;
          GSL_IMAG (*z1) = (sqrt (3.0) / 2.0) * fabs(A - B);

          GSL_REAL (*z2) = A + B - a / 3;
          GSL_IMAG (*z2) = 0;
        }

      return 3;
    }
}
