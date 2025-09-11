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

A uniform (flat) PDF in N dimensions with explicit, fittable bounds.

This class defines a probability density function that is constant within a defined
N-dimensional rectangular region and zero everywhere else. It can be used in two ways:

1.  **N-Dimensional with Fittable Bounds (Recommended):**
    By providing `RooArgSet`s for the observables, lower bounds, and upper bounds,
    you can create a fully flexible N-dimensional PDF where the boundaries of the
    uniform region are themselves fittable parameters. This is the standard way to
    model a flat background with unknown boundaries.

    Example of a 2D bounded mode:
    ```
    // Define observables and their full range
    RooRealVar x("x", "x", 0, 10);
    RooRealVar y("y", "y", 0, 10);

    // Define fittable parameters for the bounds
    RooRealVar x_low("x_low", "x_low", 2, 0, 10);
    RooRealVar x_high("x_high", "x_high", 8, 0, 10);
    RooRealVar y_low("y_low", "y_low", 3, 0, 10);
    RooRealVar y_high("y_high", "y_high", 7, 0, 10);

    // Create the 2D uniform PDF
    RooUniform model("model", "2D Bounded Uniform", {x, y}, {x_low, y_low}, {x_high, y_high});
    ```

2.  **Backward-Compatible Legacy Mode:**
    For backward compatibility, you can still call the constructor with only a `RooArgSet` of
    observables. In this mode, the PDF will be uniform over the full pre-defined range
    of each observable. This is achieved internally by creating constant bounds from each
    observable's `getMin()` and `getMax()` values.
**/

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooUniform.h"
#include "RooRandom.h"
#include "RooConstVar.h"
#include <algorithm>


////////////////////////////////////////////////////////////////////////////////
/// Legacy constructor for an N-dimensional uniform PDF.
/// This constructor creates a uniform PDF over the full range of the provided
/// observables by creating RooConstVars for the min and max of each observable
/// and delegating to the bounded constructor.

RooUniform::RooUniform(const char *name, const char *title, const RooArgSet &vars)
   : RooAbsPdf(name, title),
     _observables("observables", "List of observables", this),
     _lowerBounds("lowerBounds", "List of lower bounds", this),
     _upperBounds("upperBounds", "List of upper bounds", this)
{
   RooArgSet lowerBounds;
   RooArgSet upperBounds;
   for (const auto *var : vars) {
      const RooRealVar *rrv = static_cast<const RooRealVar *>(var);
      if (rrv) {
         lowerBounds.add(*new RooConstVar(TString::Format("%s_low", rrv->GetName()), "", rrv->getMin()));
         upperBounds.add(*new RooConstVar(TString::Format("%s_high", rrv->GetName()), "", rrv->getMax()));
      }
   }

   _observables.add(vars);
   _lowerBounds.add(lowerBounds);
   _upperBounds.add(upperBounds);
}
////////////////////////////////////////////////////////////////////////////////
/// Constructor for an N-dimensional uniform PDF with fittable bounds.
/// The number of observables, lower bounds, and upper bounds must be the same.

RooUniform::RooUniform(const char *name, const char *title, const RooArgSet& observables, const RooArgSet& lowerBounds, const RooArgSet& upperBounds) :
  RooAbsPdf(name,title),
  _observables("observables","List of observables",this),
  _lowerBounds("lowerBounds","List of lower bounds",this),
  _upperBounds("upperBounds","List of upper bounds",this)
{
  if (observables.size() != lowerBounds.size() || observables.size() != upperBounds.size()) {
    coutE(InputArguments) << "RooUniform::constructor :" << GetName() 
                          << " ERROR: Number of observables, lower bounds, and upper bounds must be the same." << std::endl;
    return;
  }

  _observables.add(observables);
  _lowerBounds.add(lowerBounds);
  _upperBounds.add(upperBounds);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooUniform::RooUniform(const RooUniform& other, const char* name) :
  RooAbsPdf(other,name),
  _observables("observables", this, other._observables),
  _lowerBounds("lowerBounds", this, other._lowerBounds),
  _upperBounds("upperBounds", this, other._upperBounds)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooUniform::evaluate() const
{
  // Loop through all dimensions
  for (unsigned int i = 0; i < _observables.size(); ++i) {
    const RooAbsReal* obs = static_cast<const RooAbsReal*>(_observables.at(i));
    const RooAbsReal* low = static_cast<const RooAbsReal*>(_lowerBounds.at(i));
    const RooAbsReal* high = static_cast<const RooAbsReal*>(_upperBounds.at(i));

    // Check for unphysical bounds in this dimension
    if (low->getVal() >= high->getVal()) return 0.0;

    // Check if the point is outside the bounds in this dimension
    if (obs->getVal() < low->getVal() || obs->getVal() > high->getVal()) {
      return 0.0;
    }
  }

  // If the point is inside the N-dimensional box, return 1.0
  return 1.0;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise analytical integral

Int_t RooUniform::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const
{
  // We can integrate over any subset of our observables
  if (matchArgs(allVars, analVars, _observables)) return 1;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integral

double RooUniform::analyticalIntegral(Int_t code, const char* rangeName) const
{
  if (code != 1) return 0.0;

  double volume = 1.0;

  // Loop through all dimensions and multiply the widths
  for (unsigned int i = 0; i < _observables.size(); ++i) {
    const RooAbsRealLValue* obs = static_cast<const RooAbsRealLValue*>(_observables.at(i));
    const RooAbsReal* low = static_cast<const RooAbsReal*>(_lowerBounds.at(i));
    const RooAbsReal* high = static_cast<const RooAbsReal*>(_upperBounds.at(i));

    if (low->getVal() >= high->getVal()) return 0.0;

    // Calculate the width of the valid integration range in this dimension
    double xmin = std::max(obs->getMin(rangeName), low->getVal());
    double xmax = std::min(obs->getMax(rangeName), high->getVal());
    
    if (xmax > xmin) {
        volume *= (xmax - xmin);
    } else {
        return 0.0; // If any dimension has zero width, the total volume is zero
    }
  }

  return volume;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise internal generator

Int_t RooUniform::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, bool /*staticInitOK*/) const
{
  if (matchArgs(directVars, generateVars, _observables)) return 1;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement internal generator

void RooUniform::generateEvent(Int_t code)
{
  if (code != 1) return;

  // Loop through all dimensions and generate a random number in each
  for (unsigned int i = 0; i < _observables.size(); ++i) {
    RooAbsRealLValue* obs = static_cast<RooAbsRealLValue*>(_observables.at(i));
    const RooAbsReal* low = static_cast<const RooAbsReal*>(_lowerBounds.at(i));
    const RooAbsReal* high = static_cast<const RooAbsReal*>(_upperBounds.at(i));
    
    if (low->getVal() < high->getVal()) {
        obs->setVal(low->getVal() + (high->getVal() - low->getVal()) * RooRandom::uniform());
    }
  }
}
