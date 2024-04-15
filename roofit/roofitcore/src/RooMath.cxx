/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [MISC] --
// RooMath is a singleton class implementing various mathematical
// functions not found in TMath, mostly involving complex algebra

#include <RooMath.h>

#include <RooHeterogeneousMath.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>

std::complex<double> RooMath::faddeeva(std::complex<double> z)
{
   return RooHeterogeneousMath::faddeeva(z);
}

std::complex<double> RooMath::faddeeva_fast(std::complex<double> z)
{
   return RooHeterogeneousMath::faddeeva_fast(z);
}

std::complex<double> RooMath::erfc(const std::complex<double> z)
{
   double re = -z.real() * z.real() + z.imag() * z.imag();
   double im = -2. * z.real() * z.imag();
   RooHeterogeneousMath::cexp(re, im);
   return (z.real() >= 0.) ? (std::complex<double>(re, im) * faddeeva(std::complex<double>(-z.imag(), z.real())))
                           : (2. - std::complex<double>(re, im) * faddeeva(std::complex<double>(z.imag(), -z.real())));
}

std::complex<double> RooMath::erfc_fast(const std::complex<double> z)
{
   double re = -z.real() * z.real() + z.imag() * z.imag();
   double im = -2. * z.real() * z.imag();
   RooHeterogeneousMath::cexp(re, im);
   return (z.real() >= 0.)
             ? (std::complex<double>(re, im) * faddeeva_fast(std::complex<double>(-z.imag(), z.real())))
             : (2. - std::complex<double>(re, im) * faddeeva_fast(std::complex<double>(z.imag(), -z.real())));
}

std::complex<double> RooMath::erf(const std::complex<double> z)
{
   double re = -z.real() * z.real() + z.imag() * z.imag();
   double im = -2. * z.real() * z.imag();
   RooHeterogeneousMath::cexp(re, im);
   return (z.real() >= 0.) ? (1. - std::complex<double>(re, im) * faddeeva(std::complex<double>(-z.imag(), z.real())))
                           : (std::complex<double>(re, im) * faddeeva(std::complex<double>(z.imag(), -z.real())) - 1.);
}

std::complex<double> RooMath::erf_fast(const std::complex<double> z)
{
   double re = -z.real() * z.real() + z.imag() * z.imag();
   double im = -2. * z.real() * z.imag();
   RooHeterogeneousMath::cexp(re, im);
   return (z.real() >= 0.)
             ? (1. - std::complex<double>(re, im) * faddeeva_fast(std::complex<double>(-z.imag(), z.real())))
             : (std::complex<double>(re, im) * faddeeva_fast(std::complex<double>(z.imag(), -z.real())) - 1.);
}

double RooMath::interpolate(double ya[], Int_t n, double x)
{
   // Interpolate array 'ya' with 'n' elements for 'x' (between 0 and 'n'-1)

   // Int to Double conversion is faster via array lookup than type conversion!
   static double itod[20] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                             10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0};
   int i;
   int m;
   int ns = 1;
   double den;
   double dif;
   double dift /*,ho,hp,w*/;
   double y;
   double dy;
   double c[20];
   double d[20];

   dif = std::abs(x);
   for (i = 1; i <= n; i++) {
      if ((dift = std::abs(x - itod[i - 1])) < dif) {
         ns = i;
         dif = dift;
      }
      c[i] = ya[i - 1];
      d[i] = ya[i - 1];
   }

   y = ya[--ns];
   for (m = 1; m < n; m++) {
      for (i = 1; i <= n - m; i++) {
         den = (c[i + 1] - d[i]) / itod[m];
         d[i] = (x - itod[i + m - 1]) * den;
         c[i] = (x - itod[i - 1]) * den;
      }
      dy = (2 * ns) < (n - m) ? c[ns + 1] : d[ns--];
      y += dy;
   }
   return y;
}

double RooMath::interpolate(double xa[], double ya[], Int_t n, double x)
{
   // Interpolate array 'ya' with 'n' elements for 'xa'

   // Int to Double conversion is faster via array lookup than type conversion!
   //   static double itod[20] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
   //                10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0} ;
   int i;
   int m;
   int ns = 1;
   double den;
   double dif;
   double dift;
   double ho;
   double hp;
   double w;
   double y;
   double dy;
   double c[20];
   double d[20];

   dif = std::abs(x - xa[0]);
   for (i = 1; i <= n; i++) {
      if ((dift = std::abs(x - xa[i - 1])) < dif) {
         ns = i;
         dif = dift;
      }
      c[i] = ya[i - 1];
      d[i] = ya[i - 1];
   }

   y = ya[--ns];
   for (m = 1; m < n; m++) {
      for (i = 1; i <= n - m; i++) {
         ho = xa[i - 1] - x;
         hp = xa[i - 1 + m] - x;
         w = c[i + 1] - d[i];
         den = ho - hp;
         if (den == 0.) {
            std::cerr << "In " << __func__ << "(), " << __FILE__ << ":" << __LINE__
                      << ": Zero distance between points not allowed." << std::endl;
            return 0;
         }
         den = w / den;
         d[i] = hp * den;
         c[i] = ho * den;
      }
      dy = (2 * ns) < (n - m) ? c[ns + 1] : d[ns--];
      y += dy;
   }
   return y;
}
