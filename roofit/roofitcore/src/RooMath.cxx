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

#include <faddeeva_impl.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>

std::complex<double> RooMath::faddeeva(std::complex<double> z)
{
   return faddeeva_impl::faddeeva(z);
}

std::complex<double> RooMath::faddeeva_fast(std::complex<double> z)
{
   return faddeeva_impl::faddeeva_fast(z);
}

std::complex<double> RooMath::erfc(const std::complex<double> z)
{
   double re = -z.real() * z.real() + z.imag() * z.imag();
   double im = -2. * z.real() * z.imag();
   faddeeva_impl::cexp(re, im);
   return (z.real() >= 0.) ? (std::complex<double>(re, im) * faddeeva(std::complex<double>(-z.imag(), z.real())))
                           : (2. - std::complex<double>(re, im) * faddeeva(std::complex<double>(z.imag(), -z.real())));
}

std::complex<double> RooMath::erfc_fast(const std::complex<double> z)
{
   double re = -z.real() * z.real() + z.imag() * z.imag();
   double im = -2. * z.real() * z.imag();
   faddeeva_impl::cexp(re, im);
   return (z.real() >= 0.)
             ? (std::complex<double>(re, im) * faddeeva_fast(std::complex<double>(-z.imag(), z.real())))
             : (2. - std::complex<double>(re, im) * faddeeva_fast(std::complex<double>(z.imag(), -z.real())));
}

std::complex<double> RooMath::erf(const std::complex<double> z)
{
   double re = -z.real() * z.real() + z.imag() * z.imag();
   double im = -2. * z.real() * z.imag();
   faddeeva_impl::cexp(re, im);
   return (z.real() >= 0.) ? (1. - std::complex<double>(re, im) * faddeeva(std::complex<double>(-z.imag(), z.real())))
                           : (std::complex<double>(re, im) * faddeeva(std::complex<double>(z.imag(), -z.real())) - 1.);
}

std::complex<double> RooMath::erf_fast(const std::complex<double> z)
{
   double re = -z.real() * z.real() + z.imag() * z.imag();
   double im = -2. * z.real() * z.imag();
   faddeeva_impl::cexp(re, im);
   return (z.real() >= 0.)
             ? (1. - std::complex<double>(re, im) * faddeeva_fast(std::complex<double>(-z.imag(), z.real())))
             : (std::complex<double>(re, im) * faddeeva_fast(std::complex<double>(z.imag(), -z.real())) - 1.);
}

Double_t RooMath::interpolate(Double_t ya[], Int_t n, Double_t x)
{
   // Interpolate array 'ya' with 'n' elements for 'x' (between 0 and 'n'-1)

   // Int to Double conversion is faster via array lookup than type conversion!
   static Double_t itod[20] = {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,
                               10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0};
   int i, m, ns = 1;
   Double_t den, dif, dift /*,ho,hp,w*/, y, dy;
   Double_t c[20], d[20];

   dif = fabs(x);
   for (i = 1; i <= n; i++) {
      if ((dift = fabs(x - itod[i - 1])) < dif) {
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

Double_t RooMath::interpolate(Double_t xa[], Double_t ya[], Int_t n, Double_t x)
{
   // Interpolate array 'ya' with 'n' elements for 'xa'

   // Int to Double conversion is faster via array lookup than type conversion!
   //   static Double_t itod[20] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
   //                10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0} ;
   int i, m, ns = 1;
   Double_t den, dif, dift, ho, hp, w, y, dy;
   Double_t c[20], d[20];

   dif = fabs(x - xa[0]);
   for (i = 1; i <= n; i++) {
      if ((dift = fabs(x - xa[i - 1])) < dif) {
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
