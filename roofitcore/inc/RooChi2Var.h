/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooChi2Var.rdl,v 1.11 2005/02/25 14:22:54 wverkerke Exp $
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

#ifndef ROO_CHI2_VAR
#define ROO_CHI2_VAR

#include "RooAbsOptGoodnessOfFit.h"
#include "RooCmdArg.h"
#include "RooDataHist.h"

class RooChi2Var : public RooAbsOptGoodnessOfFit {
public:

  // Constructors, assignment etc
  RooChi2Var(const char *name, const char* title, RooAbsPdf& pdf, RooDataHist& data,
	     const RooCmdArg& arg1                , const RooCmdArg& arg2=RooCmdArg::none,const RooCmdArg& arg3=RooCmdArg::none,
	     const RooCmdArg& arg4=RooCmdArg::none, const RooCmdArg& arg5=RooCmdArg::none,const RooCmdArg& arg6=RooCmdArg::none,
	     const RooCmdArg& arg7=RooCmdArg::none, const RooCmdArg& arg8=RooCmdArg::none,const RooCmdArg& arg9=RooCmdArg::none) ;

  RooChi2Var(const char *name, const char *title, RooAbsPdf& pdf, RooDataHist& data,
	    Bool_t extended=kFALSE, const char* rangeName=0, Int_t nCPU=1, Bool_t verbose=kTRUE, Bool_t splitCutRange=kTRUE) ;

  RooChi2Var(const char *name, const char *title, RooAbsPdf& pdf, RooDataHist& data,
	    const RooArgSet& projDeps, Bool_t extended=kFALSE, const char* rangeName=0, Int_t nCPU=1, Bool_t verbose=kTRUE, Bool_t splitCutRange=kTRUE) ;

  RooChi2Var(const RooChi2Var& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooChi2Var(*this,newname); }

  virtual RooAbsGoodnessOfFit* create(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
				      const RooArgSet& projDeps, const char* rangeName=0, Int_t nCPU=1, Bool_t verbose=kTRUE, Bool_t splitCutRange=kTRUE) {
    return new RooChi2Var(name,title,pdf,(RooDataHist&)data,projDeps,_extended,rangeName,nCPU, verbose, splitCutRange) ;
  }
  
  virtual ~RooChi2Var();

  virtual Double_t defaultErrorLevel() const { return 1.0 ; }

protected:

  static RooArgSet _emptySet ; // Supports named argument constructor

  RooDataHist::ErrorType _etype ;
  Bool_t _extended ;
  virtual Double_t evaluatePartition(Int_t firstEvent, Int_t lastEvent) const ;
  
  ClassDef(RooChi2Var,1) // Abstract real-valued variable
};


#endif
