/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealAnalytic.rdl,v 1.1 2001/08/03 21:44:57 david Exp $
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
  inline RooRealAnalytic(const RooAbsReal &func, const RooArgSet &vars, Int_t code, const RooArgSet* normSet=0) :
    RooRealBinding(func,vars,normSet), _code(code) { }
  inline virtual ~RooRealAnalytic() { }

  virtual Double_t operator()(const Double_t xvector[]) const;

protected:
  Int_t _code;

  ClassDef(RooRealAnalytic,0) // RooAbsFunc decorator
};

#endif

