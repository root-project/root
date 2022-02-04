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

#ifndef ROOPROJECTEDPDF
#define ROOPROJECTEDPDF

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
#include "RooObjCacheManager.h"
#include "RooSetProxy.h"

class RooProjectedPdf : public RooAbsPdf {
public:

  RooProjectedPdf() ;
  RooProjectedPdf(const char *name, const char *title,  RooAbsReal& _intpdf, const RooArgSet& intObs);
  RooProjectedPdf(const RooProjectedPdf& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooProjectedPdf(*this,newname); }
  inline ~RooProjectedPdf() override { }

  // Analytical integration support
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const override ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const override ;
  Bool_t forceAnalyticalInt(const RooAbsArg& dep) const override ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const override;
  void initGenerator(Int_t /*code*/) override {} ; // optional pre-generation initialization
  void generateEvent(Int_t code) override;

  Bool_t selfNormalized() const override { return kTRUE ; }

  // Handle projection of projection explicitly
  RooAbsPdf* createProjection(const RooArgSet& iset) override ;

  void printMetaArgs(std::ostream& os) const override ;


protected:

  RooRealProxy intpdf ; ///< p.d.f that is integrated
  RooSetProxy intobs ;  ///< observables that p.d.f is integrated over
  RooSetProxy deps ;    ///< dependents of this p.d.f

  class CacheElem : public RooAbsCacheElement {
  public:
    ~CacheElem() override { delete _projection ; } ;
    // Payload
    RooAbsReal* _projection ;
    // Cache management functions
    RooArgList containedArgs(Action) override ;
    void printCompactTreeHook(std::ostream&, const char *, Int_t, Int_t) override ;
  } ;
  mutable RooObjCacheManager _cacheMgr ; ///<! The cache manager

  Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) override ;

  const RooAbsReal* getProjection(const RooArgSet* iset, const RooArgSet* nset, const char* rangeName, int& code) const ;
  Double_t evaluate() const override ;

private:

  ClassDefOverride(RooProjectedPdf,1) // Operator p.d.f calculating projection of another p.d.f
};

#endif
