/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealBinding.rdl,v 1.1 2001/08/03 21:44:57 david Exp $
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
  RooRealBinding(const RooAbsReal& func, const RooArgSet &vars, const RooArgSet* nset=0);
  virtual ~RooRealBinding();

  virtual Double_t operator()(const Double_t xvector[]) const;
  virtual Double_t getMinLimit(UInt_t dimension) const;
  virtual Double_t getMaxLimit(UInt_t dimension) const;

protected:
  void loadValues(const Double_t xvector[]) const;
  const RooAbsReal *_func;
  RooAbsRealLValue **_vars;
  const RooArgSet *_nset;

  ClassDef(RooRealBinding,0) // RooAbsReal interface adaptor
};

#endif

