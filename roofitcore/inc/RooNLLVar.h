/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNLLVar.rdl,v 1.8 2005/02/25 14:22:59 wverkerke Exp $
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
#ifndef ROO_NLL_VAR
#define ROO_NLL_VAR

#include "RooAbsOptGoodnessOfFit.h"
#include "RooCmdArg.h"

class RooNLLVar : public RooAbsOptGoodnessOfFit {
public:

  // Constructors, assignment etc
  RooNLLVar(const char *name, const char* title, RooAbsPdf& pdf, RooAbsData& data,
	    const RooCmdArg& arg1                , const RooCmdArg& arg2=RooCmdArg::none,const RooCmdArg& arg3=RooCmdArg::none,
	    const RooCmdArg& arg4=RooCmdArg::none, const RooCmdArg& arg5=RooCmdArg::none,const RooCmdArg& arg6=RooCmdArg::none,
	    const RooCmdArg& arg7=RooCmdArg::none, const RooCmdArg& arg8=RooCmdArg::none,const RooCmdArg& arg9=RooCmdArg::none) ;

  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
	    Bool_t extended=kFALSE, const char* rangeName=0, Int_t nCPU=1, Bool_t verbose=kTRUE, Bool_t splitRange=kFALSE) ;
  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
	    const RooArgSet& projDeps, Bool_t extended=kFALSE, const char* rangeName=0, Int_t nCPU=1, Bool_t verbose=kTRUE, Bool_t splitRange=kFALSE) ;
  RooNLLVar(const RooNLLVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooNLLVar(*this,newname); }

  virtual RooAbsGoodnessOfFit* create(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
				      const RooArgSet& projDeps, const char* rangeName, Int_t nCPU=1, Bool_t verbose=kTRUE, Bool_t splitRange=kFALSE) {
    return new RooNLLVar(name,title,pdf,data,projDeps,_extended,rangeName, nCPU, verbose,splitRange) ;
  }
  
  virtual ~RooNLLVar();

  virtual Double_t defaultErrorLevel() const { return 0.5 ; }

protected:

  static RooArgSet _emptySet ; // Supports named argument constructor

  Bool_t _extended ;
  virtual Double_t evaluatePartition(Int_t firstEvent, Int_t lastEvent) const ;
  
  ClassDef(RooNLLVar,1) // Abstract real-valued variable
};

#endif
