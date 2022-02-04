// Author: Rahul Balasubramanian, Nikhef 08 Apr 2021
/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsReal.h,v 1.75 2007/07/13 21:50:24 wouter Exp $
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
#ifndef ROO_LINEAR_COMB
#define ROO_LINEAR_COMB

#include "RooFit/Floats.h"

#include <list>
#include <ostream>
#include <vector>

#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooListProxy.h"

class RooLinearCombination : public RooAbsReal {
  RooListProxy _actualVars;
  std::vector<RooFit::SuperFloat> _coefficients;
  mutable RooArgSet *_nset; //!

public:
  RooLinearCombination();
  RooLinearCombination(const char *name);
  RooLinearCombination(const RooLinearCombination &other, const char *name);
  void printArgs(std::ostream &os) const override;
  ~RooLinearCombination() override;
  TObject *clone(const char *newname) const override;
  void add(RooFit::SuperFloat c, RooAbsReal *t);
  void setCoefficient(size_t idx, RooFit::SuperFloat c);
  RooFit::SuperFloat getCoefficient(size_t idx);
  Double_t evaluate() const override;
  std::list<Double_t> *binBoundaries(RooAbsRealLValue &obs,
                                             Double_t xlo,
                                             Double_t xhi) const override;
  std::list<Double_t> *plotSamplingHint(RooAbsRealLValue &obs,
                                                Double_t xlo,
                                                Double_t xhi) const override;

  ClassDefOverride(RooLinearCombination, 1)
};

#endif
