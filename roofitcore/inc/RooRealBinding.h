/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsFunc1D.rdl,v 1.2 2001/05/14 22:54:19 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   03-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/
#ifndef ROO_REAL_BINDING
#define ROO_REAL_BINDING

#include "RooFitCore/RooAbsFunc.hh"

class RooAbsRealLValue;
class RooAbsReal;
class RooArgSet;

class RooRealBinding : public RooAbsFunc {
public:
  RooRealBinding(const RooAbsReal& func, const RooArgSet &vars);
  virtual ~RooRealBinding();

  virtual Double_t operator()(const Double_t xvector[]) const;
  virtual Double_t getMinLimit(UInt_t dimension) const;
  virtual Double_t getMaxLimit(UInt_t dimension) const;

protected:
  void loadValues(const Double_t xvector[]) const;
  const RooAbsReal *_func;
  RooAbsRealLValue **_vars;

  ClassDef(RooRealBinding,0) // RooAbsReal interface adaptor
};

#endif

