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
#include "RooNormListManager.h"
#include "RooSetProxy.h" 

class RooProjectedPdf : public RooAbsPdf {
public:

  RooProjectedPdf(const char *name, const char *title,  RooAbsReal& _intpdf, const RooArgSet& intObs);
  RooProjectedPdf(const RooProjectedPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooProjectedPdf(*this,newname); }
  inline virtual ~RooProjectedPdf() { }

  // Analytical integration support
  virtual Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const ;
  virtual Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;
  virtual Bool_t forceAnalyticalInt(const RooAbsArg& dep) const ;

  virtual Double_t getVal(const RooArgSet* set=0) const ;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void initGenerator(Int_t /*code*/) {} ; // optional pre-generation initialization
  void generateEvent(Int_t code);

  virtual Bool_t selfNormalized() const { return kTRUE ; }

  // Handle projection of projection explicitly
  virtual RooAbsPdf* createProjection(const RooArgSet& iset) ;  


protected:

  RooRealProxy intpdf ; // p.d.f that is integrated
  RooSetProxy intobs ;  // observables that p.d.f is integrated over
  RooSetProxy deps ;    // dependents of this p.d.f

  mutable RooNormListManager _projListMgr ; // Partial integral list manager
  mutable RooArgSet* _curNormSet ; //!

  void clearCache() ;  
  virtual void operModeHook() ;
  virtual Bool_t redirectServersHook(const RooAbsCollection& newServerList, Bool_t mustReplaceAll, Bool_t nameChange, Bool_t isRecursive) ;
  virtual void printCompactTreeHook(ostream& os, const char* indent="") ;


  const RooAbsReal* getProjection(const RooArgSet* iset, const RooArgSet* nset, int& code) const ;
  Double_t evaluate() const ;

private:

  ClassDef(RooProjectedPdf,0) // Your description goes here...
};
 
#endif
