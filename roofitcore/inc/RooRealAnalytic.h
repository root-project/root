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
#ifndef ROO_REAL_ANALYTIC
#define ROO_REAL_ANALYTIC

#include "RooFitCore/RooRealBinding.hh"

class RooRealAnalytic : public RooRealBinding {
public:
  inline RooRealAnalytic(const RooAbsReal &func, const RooArgSet &vars, Int_t code) :
    RooRealBinding(func,vars), _code(code) { }
  inline virtual ~RooRealAnalytic() { }

  virtual Double_t operator()(const Double_t xvector[]) const;

protected:
  Int_t _code;

  ClassDef(RooRealAnalytic,0) // RooAbsFunc decorator
};

#endif

