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

/**
\file RooBrentRootFinder.cxx
\class RooBrentRootFinder
\ingroup Roofitcore

Implement the abstract 1-dimensional root finding interface using
the Brent-Decker method. This implementation is based on the one
in the GNU scientific library (v0.99).
**/

#include "RooBrentRootFinder.h"
#include "RooAbsFunc.h"
#include <math.h>
#include "Riostream.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooBrentRootFinder);
;


////////////////////////////////////////////////////////////////////////////////
/// Constructor taking function binding as input

RooBrentRootFinder::RooBrentRootFinder(const RooAbsFunc& function) :
  RooAbsRootFinder(function),
  _tol(2.2204460492503131e-16)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Do the root finding using the Brent-Decker method. Returns a boolean status and
/// loads 'result' with our best guess at the root if true.
/// Prints a warning if the initial interval does not bracket a single
/// root or if the root is not found after a fixed number of iterations.

bool RooBrentRootFinder::findRoot(double &result, double xlo, double xhi, double value) const
{
  _function->saveXVec() ;

  double a(xlo),b(xhi);
  double fa= (*_function)(&a) - value;
  double fb= (*_function)(&b) - value;
  if(fb*fa > 0) {
    oocxcoutD((TObject*)0,Eval) << "RooBrentRootFinder::findRoot(" << _function->getName() << "): initial interval does not bracket a root: ("
            << a << "," << b << "), value = " << value << " f[xlo] = " << fa << " f[xhi] = " << fb << endl;
    return false;
  }

  bool ac_equal(false);
  double fc= fb;
  double c(0),d(0),e(0);
  for(Int_t iter= 0; iter <= MaxIterations; iter++) {

    if ((fb < 0 && fc < 0) || (fb > 0 && fc > 0)) {
      // Rename a,b,c and adjust bounding interval d
      ac_equal = true;
      c = a;
      fc = fa;
      d = b - a;
      e = b - a;
    }

    if (std::abs(fc) < std::abs(fb)) {
      ac_equal = true;
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }

    double tol = 0.5 * _tol * std::abs(b);
    double m = 0.5 * (c - b);


    if (fb == 0 || std::abs(m) <= tol) {
      //cout << "RooBrentRootFinder: iter = " << iter << " m = " << m << " tol = " << tol << endl ;
      result= b;
      _function->restoreXVec() ;
      return true;
    }

    if (std::abs(e) < tol || std::abs(fa) <= std::abs(fb)) {
      // Bounds decreasing too slowly: use bisection
      d = m;
      e = m;
    }
    else {
      // Attempt inverse cubic interpolation
      double p, q, r;
      double s = fb / fa;

      if (ac_equal) {
   p = 2 * m * s;
   q = 1 - s;
      }
      else {
   q = fa / fc;
   r = fb / fc;
   p = s * (2 * m * q * (q - r) - (b - a) * (r - 1));
   q = (q - 1) * (r - 1) * (s - 1);
      }
      // Check whether we are in bounds
      if (p > 0) {
   q = -q;
      }
      else {
   p = -p;
      }

      double min1= 3 * m * q - std::abs(tol * q);
      double min2= std::abs(e * q);
      if (2 * p < (min1 < min2 ? min1 : min2)) {
   // Accept the interpolation
   e = d;
   d = p / q;
      }
      else {
   // Interpolation failed: use bisection.
   d = m;
   e = m;
      }
    }
    // Move last best guess to a
    a = b;
    fa = fb;
    // Evaluate new trial root
    if (std::abs(d) > tol) {
      b += d;
    }
    else {
      b += (m > 0 ? +tol : -tol);
    }
    fb= (*_function)(&b) - value;

  }
  // Return our best guess if we run out of iterations
  oocoutE(nullptr,Eval) << "RooBrentRootFinder::findRoot(" << _function->getName() << "): maximum iterations exceeded." << endl;
  result= b;

  _function->restoreXVec() ;

  return false;
}
