/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOPROFILELL
#define ROOPROFILELL

#include "RooAbsReal.h"
#include "RooMinimizer.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include <map>
#include <string>

class RooProfileLL : public RooAbsReal {
public:

  RooProfileLL() ;
  RooProfileLL(const char *name, const char *title, RooAbsReal& nll, const RooArgSet& observables);
  RooProfileLL(const RooProfileLL& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooProfileLL(*this,newname); }

  void setAlwaysStartFromMin(Bool_t flag) { _startFromMin = flag ; }
  Bool_t alwaysStartFromMin() const { return _startFromMin ; }

  RooMinimizer* minimizer() { if (!_minimizer) initializeMinimizer(); return _minimizer.get() ; }
  RooAbsReal& nll() { return const_cast<RooAbsReal&>(_nll.arg()) ; }
  const RooArgSet& bestFitParams() const ;
  const RooArgSet& bestFitObs() const ;

  RooAbsReal* createProfile(const RooArgSet& paramsOfInterest) override ;

  Bool_t redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) override ;

  void clearAbsMin() { _absMinValid = kFALSE ; }

  Int_t numEval() const { return _neval ; }


protected:

  void validateAbsMin() const ;
  void initializeMinimizer() const ;

  RooRealProxy _nll ;    ///< Input -log(L) function
  RooSetProxy _obs ;     ///< Parameters of profile likelihood
  RooSetProxy _par ;     ///< Marginalised parameters of likelihood
  Bool_t _startFromMin ; ///< Always start minimization for global minimum?

  TIterator* _piter ; ///<! Iterator over profile likelihood parameters to be minimized
  TIterator* _oiter ; ///<! Iterator of profile likelihood output parameter(s)

  mutable std::unique_ptr<RooMinimizer> _minimizer = nullptr ; ///<! Internal minimizer instance

  mutable Bool_t _absMinValid ; ///< flag if absmin is up-to-date
  mutable Double_t _absMin ; ///< absolute minimum of -log(L)
  mutable RooArgSet _paramAbsMin ; ///< Parameter values at absolute minimum
  mutable RooArgSet _obsAbsMin ; ///< Observable values at absolute minimum
  mutable std::map<std::string,bool> _paramFixed ; ///< Parameter constant status at last time of use
  mutable Int_t _neval ; ///< Number evaluations used in last minimization
  Double_t evaluate() const override ;


private:

  ClassDefOverride(RooProfileLL,0) // Real-valued function representing profile likelihood of external (likelihood) function
};

#endif
