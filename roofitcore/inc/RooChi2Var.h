/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooChi2Var.rdl,v 1.4 2003/04/07 21:39:15 wverkerke Exp $
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

#ifndef ROO_CHI2_VAR
#define ROO_CHI2_VAR

#include "RooFitCore/RooAbsOptGoodnessOfFit.hh"
#include "RooFitCore/RooCmdArg.hh"
#include "RooFitCore/RooDataHist.hh"

class RooChi2Var : public RooAbsOptGoodnessOfFit {
public:

  // Constructors, assignment etc
  RooChi2Var(const char *name, const char* title, RooAbsPdf& pdf, RooDataHist& data,
	     const RooCmdArg& arg1=RooCmdArg(), const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),
	     const RooCmdArg& arg4=RooCmdArg(), const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),
	     const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg(),const RooCmdArg& arg9=RooCmdArg()) ;

  RooChi2Var(const char *name, const char *title, RooAbsPdf& pdf, RooDataHist& data,
	    Bool_t extended, Int_t nCPU=1) ;

  RooChi2Var(const char *name, const char *title, RooAbsPdf& pdf, RooDataHist& data,
	    const RooArgSet& projDeps, Bool_t extended=kFALSE, Int_t nCPU=1) ;

  RooChi2Var(const RooChi2Var& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooChi2Var(*this,newname); }

  virtual RooAbsGoodnessOfFit* create(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
				      const RooArgSet& projDeps, Int_t nCPU=1) {
    return new RooChi2Var(name,title,pdf,(RooDataHist&)data,projDeps,_extended,nCPU) ;
  }
  
  virtual ~RooChi2Var();

  virtual Double_t defaultErrorLevel() const { return 1.0 ; }

protected:

  RooDataHist::ErrorType _etype ;
  Bool_t _extended ;
  virtual Double_t evaluatePartition(Int_t firstEvent, Int_t lastEvent) const ;
  
  ClassDef(RooChi2Var,1) // Abstract real-valued variable
};

RooCmdArg Extended() ;
RooCmdArg DataError(RooDataHist::ErrorType) ;
RooCmdArg NumCPU(Int_t nCPU) ;


#endif
