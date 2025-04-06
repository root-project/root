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
  RooProjectedPdf(const RooProjectedPdf& other, const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooProjectedPdf(*this,newname); }

  // Analytical integration support
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;
  bool forceAnalyticalInt(const RooAbsArg& dep) const override ;

  void initGenerator(Int_t /*code*/) override {} ; // optional pre-generation initialization

  bool selfNormalized() const override { return true ; }

  // Handle projection of projection explicitly
  RooAbsPdf* createProjection(const RooArgSet& iset) override ;

  void printMetaArgs(std::ostream& os) const override ;

  std::unique_ptr<RooAbsArg> compileForNormSet(RooArgSet const &normSet, RooFit::Detail::CompileContext & ctx) const override;

  // Handle case of projecting an Extended pdf
  double expectedEvents(const RooArgSet* nset) const override { return static_cast<RooAbsPdf*>(intpdf.absArg())->expectedEvents(nset); }
  ExtendMode extendMode() const override { return static_cast<RooAbsPdf*>(intpdf.absArg())->extendMode(); }
  

protected:

  RooRealProxy intpdf ; ///< p.d.f that is integrated
  RooSetProxy intobs ;  ///< observables that p.d.f is integrated over
  RooSetProxy deps ;    ///< dependents of this p.d.f

  class CacheElem : public RooAbsCacheElement {
  public:
    // Payload
    std::unique_ptr<RooAbsReal> _projection;
    // Cache management functions
    RooArgList containedArgs(Action) override ;
    void printCompactTreeHook(std::ostream&, const char *, Int_t, Int_t) override ;
  } ;
  mutable RooObjCacheManager _cacheMgr ; ///<! The cache manager

  bool redirectServersHook(const RooAbsCollection& newServerList, bool /*mustReplaceAll*/, bool /*nameChange*/, bool /*isRecursive*/) override ;

  const RooAbsReal* getProjection(const RooArgSet* iset, const RooArgSet* nset, const char* rangeName, int& code) const ;
  double evaluate() const override ;

private:

  ClassDefOverride(RooProjectedPdf,1) // Operator p.d.f calculating projection of another p.d.f
};

#endif
