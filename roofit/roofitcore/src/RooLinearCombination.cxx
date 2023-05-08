// Author: Rahul Balasubramanian, Nikhef 08 Apr 2021
/*****************************************************************************
 * RooFit
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2019, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/// \class RooLinearCombination
/// RooLinearCombination is a class that helps perform linear combination of
/// floating point numbers and permits handling them as multiprecision
///

#include "RooLinearCombination.h"

#include "Math/Util.h"

ClassImp(RooLinearCombination);

namespace {
  template <class T> inline void assign(RooFit::SuperFloat &var, const T &val) {
  #ifdef USE_UBLAS
    var.assign(val);
  #else
    var = val;
  #endif
  }
} // namespace

RooLinearCombination::RooLinearCombination()
    : _actualVars("actualVars", "Variables used by formula expression", this),
      _nset(0) {
  // constructor
}

RooLinearCombination::RooLinearCombination(const char *name)
    : RooAbsReal(name, name),
      _actualVars("actualVars", "Variables used by formula expression", this),
      _nset(0) {
  // constructor
}

RooLinearCombination::RooLinearCombination(const RooLinearCombination &other,
                                     const char *name)
    : RooAbsReal(other, name),
      _actualVars("actualVars", this, other._actualVars),
      _coefficients(other._coefficients), _nset(0) {
  // copy constructor
}

void RooLinearCombination::printArgs(std::ostream &os) const {
  // detailed printing method
  os << "[";
  const std::size_t n(_actualVars.getSize());
  for (std::size_t i = 0; i < n; ++i) {
    const RooAbsReal *r =
        static_cast<const RooAbsReal *>(_actualVars.at(i));
    double c(_coefficients[i]);
    if (c > 0 && i > 0)
      os << "+";
    os << c << "*" << r->GetTitle();
  }
  os << "]";
}

RooLinearCombination::~RooLinearCombination() {
  // destructor
}

TObject *RooLinearCombination::clone(const char *newname) const {
  // create a clone (deep copy) of this object
  RooLinearCombination *retval = new RooLinearCombination(newname);
  const std::size_t n(_actualVars.getSize());
  for (std::size_t i = 0; i < n; ++i) {
    const RooAbsReal *r =
        static_cast<const RooAbsReal *>(_actualVars.at(i));
    retval->add(_coefficients[i], static_cast<RooAbsReal *>(r->clone()));
  }
  return retval;
}

void RooLinearCombination::add(RooFit::SuperFloat c, RooAbsReal *t) {
  // add a new term
  _actualVars.add(*t);
  _coefficients.push_back(c);
}

void RooLinearCombination::setCoefficient(size_t idx, RooFit::SuperFloat c) {
  // set the coefficient with the given index
  _coefficients[idx] = c;
}

RooFit::SuperFloat RooLinearCombination::getCoefficient(size_t idx) {
  // get the coefficient with the given index
  return _coefficients[idx];
}

double RooLinearCombination::evaluate() const {
  // call the evaluation
#ifdef USE_UBLAS
    RooFit::SuperFloat result;
  result.assign(0.0);
  const std::size_t n(_actualVars.getSize());
  for (std::size_t i = 0; i < n; ++i) {
      RooFit::SuperFloat tmp;
    tmp.assign(static_cast<const RooAbsReal *>(_actualVars.at(i))->getVal());
    result += _coefficients[i] * tmp;
  }
  return result.convert_to<double>();
#else
  const std::size_t n(_actualVars.getSize());
  std::vector<double> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = _coefficients[i] * static_cast<const RooAbsReal *>(_actualVars.at(i))->getVal();
  }
  // the values might span multiple orders of magnitudes, and to minimize
  // precision loss, we sum up the values from the smallest to the largest
  // absolute value.
  std::sort(values.begin(), values.end(), [](double const& x, double const& y){ return std::abs(x) < std::abs(y); });
  return ROOT::Math::KahanSum<double>::Accumulate(values.begin(), values.end()).Sum();
#endif
}

std::list<double> *RooLinearCombination::binBoundaries(RooAbsRealLValue &obs,
                                                      double xlo,
                                                      double xhi) const {
  // Forward the plot sampling hint from the p.d.f. that defines the observable
  // obs
  for(auto const& func : _actualVars) {
    auto binb = static_cast<RooAbsReal*>(func)->binBoundaries(obs, xlo, xhi);
    if (binb) {
      return binb;
    }
  }
  return 0;
}

std::list<double> *RooLinearCombination::plotSamplingHint(RooAbsRealLValue &obs,
                                                         double xlo,
                                                         double xhi) const {
  // Forward the plot sampling hint from the p.d.f. that defines the observable
  // obs
  for(auto const& func : _actualVars) {
    auto hint = static_cast<RooAbsReal*>(func)->plotSamplingHint(obs, xlo, xhi);
    if (hint) {
      return hint;
    }
  }
  return 0;
}
