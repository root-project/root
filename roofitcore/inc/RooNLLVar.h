/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_NLL_VAR
#define ROO_NLL_VAR

#include "RooFitCore/RooAbsOptGoodnessOfFit.hh"

class RooNLLVar : public RooAbsOptGoodnessOfFit {
public:

  // Constructors, assignment etc
  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
	    Bool_t extended=kFALSE, Int_t nCPU=1) ;
  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
	    const RooArgSet& projDeps, Bool_t extended=kFALSE, Int_t nCPU=1) ;
  RooNLLVar(const RooNLLVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooNLLVar(*this,newname); }

  virtual RooAbsGoodnessOfFit* create(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
				      const RooArgSet& projDeps, Int_t nCPU=1) {
    return new RooNLLVar(name,title,pdf,data,projDeps,_extended,nCPU) ;
  }
  
  virtual ~RooNLLVar();

  virtual Double_t defaultErrorLevel() const { return 0.5 ; }

protected:

  Bool_t _extended ;
  virtual Double_t evaluatePartition(Int_t firstEvent, Int_t lastEvent) const ;
  
  ClassDef(RooNLLVar,1) // Abstract real-valued variable
};

#endif
