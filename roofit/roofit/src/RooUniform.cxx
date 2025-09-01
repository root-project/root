/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
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

/** \class RooUniform
    \ingroup Roofit

Flat p.d.f. in N dimensions or in 1D with explicit, fittable bounds.

This class can be used in two ways:
1.  **Multi-dimensional (Legacy):** By providing a RooArgSet of observables in the constructor, it creates
    a PDF that is uniform over the full range of all observables. This is the original behavior.

2.  **1D with Fittable Bounds:** By providing a single observable (`x`) and two RooAbsReal
    parameters for the lower (`x_low`) and upper (`x_up`) bounds, it creates a 1D PDF that is uniform
    only between those bounds and zero everywhere else.

    Example of the 1D bounded mode:
    ```
    RooRealVar x("x", "x", 0, 10);
    RooRealVar low("low", "low", 2, 0, 10);
    RooRealVar high("high", "high", 8, 0, 10);
    RooUniform bounded_uniform("bounded", "bounded", x, low, high);
    ```
**/

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooUniform.h"
#include "RooRandom.h"
#include <algorithm>

ClassImp(RooUniform);

////////////////////////////////////////////////////////////////////////////////
/// Legacy constructor for an N-dimensional uniform PDF.

RooUniform::RooUniform(const char *name, const char *title, const RooArgSet &vars)
   : RooAbsPdf(name, title),
     x("x", "Observables", this, true, false),
     x_single("x_single", "", this),
     x_low("x_low", "", this),
     x_up("x_up", "", this)
{
   x.add(vars);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for a 1D uniform PDF with fittable bounds.

RooUniform::RooUniform(const char *name, const char *title, RooAbsReal &var, RooAbsReal &low, RooAbsReal &up)
   : RooAbsPdf(name, title),
     x("x", "Observables", this, true, false),
     x_single("x_single", "Observable", this, var),
     x_low("x_low", "Lower Bound", this, low),
     x_up("x_up", "Upper Bound", this, up)
{
   // Add the single observable to the list proxy for compatibility with legacy code
   x.add(var);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooUniform::RooUniform(const RooUniform &other, const char *name)
   : RooAbsPdf(other, name),
     x("x", this, other.x),
     x_single("x_single", this, other.x_single),
     x_low("x_low", this, other.x_low),
     x_up("x_up", this, other.x_up)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooUniform::evaluate() const
{
   // If 1D bounded mode (check if proxies are valid)
   if (x_single.absArg() && x_low.absArg() && x_up.absArg()) {
      double low = x_low;
      double up = x_up;

      if (low >= up)
         return 0.0; // Unphysical bounds

      double val = x_single;
      if (val >= low && val <= up) {
         // Return the correctly normalized value for a uniform PDF
         return 1.0;
      } else {
         return 0.0;
      }
   }
   // uniform over observable range
   return 1.0;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise analytical integral

Int_t RooUniform::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   // 1D explicit bounds mode
   if (x_single.absArg() && x_low.absArg() && x_up.absArg()) {
      if (matchArgs(allVars, analVars, x_single))
         return 1;
      return 0;
   }
   // multi-dimensional mode
   Int_t nx = x.size();
   if (nx > 31) {
      coutW(Integration) << "RooUniform::getAnalyticalIntegral(" << GetName() << ") WARNING: p.d.f. has " << x.size()
                         << " observables, analytical integration is only implemented for the first 31 observables"
                         << std::endl;
      nx = 31;
   }
   Int_t code(0);
   for (std::size_t i = 0; i < x.size(); i++) {
      if (allVars.find(x.at(i)->GetName())) {
         code |= (1 << i);
         analVars.add(*allVars.find(x.at(i)->GetName()));
      }
   }
   return code;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integral

double RooUniform::analyticalIntegral(Int_t code, const char *rangeName) const
{
   // 1D explicit bounds mode
   if (code == 1 && x_single.absArg() && x_low.absArg() && x_up.absArg()) {
      const RooAbsRealLValue &var = static_cast<const RooAbsRealLValue &>(x_single.arg());
      double low = x_low;
      double up = x_up;

      if (low >= up)
         return 0.0;

      double xmin = std::max(var.getMin(rangeName), low);
      double xmax = std::min(var.getMax(rangeName), up);

      if (xmax > xmin)
         return (xmax - xmin);
      return 0.0;
   }

   // multi-dimensional mode
   double ret(1);
   for (int i = 0; i < 32; i++) {
      if (code & (1 << i)) {
         RooAbsRealLValue *var = static_cast<RooAbsRealLValue *>(x.at(i));
         ret *= (var->getMax(rangeName) - var->getMin(rangeName));
      }
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise internal generator

Int_t RooUniform::getGenerator(const RooArgSet &directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
   if (x_single.absArg() && x_low.absArg() && x_up.absArg()) {
      if (matchArgs(directVars, generateVars, x_single))
         return 2;
      return 0;
   }
   Int_t nx = x.size();
   if (nx > 31) {
      // Warn that analytical integration is only provided for the first 31 observables
      coutW(Integration) << "RooUniform::getGenerator(" << GetName() << ") WARNING: p.d.f. has " << x.size()
                         << " observables, internal integrator is only implemented for the first 31 observables"
                         << std::endl;
      nx = 31;
   }

   Int_t code(0);
   for (std::size_t i = 0; i < x.size(); i++) {
      if (directVars.find(x.at(i)->GetName())) {
         code |= (1 << i);
         generateVars.add(*directVars.find(x.at(i)->GetName()));
      }
   }
   return code;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement internal generator

void RooUniform::generateEvent(Int_t code)
{
   if (code == 2) { // 1D Bounded case
      double low = x_low;
      double up = x_up;
      if (low < up) {
         static_cast<RooAbsRealLValue *>(x_single.absArg())->setVal(low + (up - low) * RooRandom::uniform());
      }
      return;
   }

   // multi-dimensional case

   // Fast-track handling of one-observable case
   if (code == 1) {
      (static_cast<RooAbsRealLValue *>(x.at(0)))->randomize();
      return;
   }
   
   for (int i = 0; i < 32; i++) {
      if (code & (1 << i)) {
         RooAbsRealLValue *var = static_cast<RooAbsRealLValue *>(x.at(i));
         var->randomize();
      }
   }
}
