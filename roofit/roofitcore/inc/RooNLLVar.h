/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNLLVar.h,v 1.10 2007/07/21 21:32:52 wouter Exp $
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

#include "RooAbsOptTestStatistic.h"
#include "RooCmdArg.h"
#include "RooAbsPdf.h"

class RooNLLVar : public RooAbsOptTestStatistic {
public:

  // Constructors, assignment etc
  RooNLLVar() { _first = kTRUE ; }
  RooNLLVar(const char *name, const char* title, RooAbsPdf& pdf, RooAbsData& data,
	    const RooCmdArg& arg1                , const RooCmdArg& arg2=RooCmdArg::none(),const RooCmdArg& arg3=RooCmdArg::none(),
	    const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none(),
	    const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none(),const RooCmdArg& arg9=RooCmdArg::none()) ;

  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
	    Bool_t extended=kFALSE, const char* rangeName=0, const char* addCoefRangeName=0, 
	    Int_t nCPU=1, Bool_t interleave=kFALSE, Bool_t verbose=kTRUE, Bool_t splitRange=kFALSE, Bool_t cloneData=kTRUE) ;

  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
	    const RooArgSet& projDeps, Bool_t extended=kFALSE, const char* rangeName=0, 
	    const char* addCoefRangeName=0, Int_t nCPU=1, Bool_t interleave=kFALSE, Bool_t verbose=kTRUE, Bool_t splitRange=kFALSE, Bool_t cloneData=kTRUE) ;

  RooNLLVar(const RooNLLVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooNLLVar(*this,newname); }

  virtual RooAbsTestStatistic* create(const char *name, const char *title, RooAbsReal& pdf, RooAbsData& adata,
				      const RooArgSet& projDeps, const char* rangeName, const char* addCoefRangeName=0, 
				      Int_t nCPU=1, Bool_t interleave=kFALSE, Bool_t verbose=kTRUE, Bool_t splitRange=kFALSE) {
    return new RooNLLVar(name,title,(RooAbsPdf&)pdf,adata,projDeps,_extended,rangeName, addCoefRangeName, nCPU, interleave,verbose,splitRange,kFALSE) ;
  }
  
  virtual ~RooNLLVar();

  void applyWeightSquared(Bool_t flag) { _weightSq = flag ; setValueDirty() ; } 

  virtual Double_t defaultErrorLevel() const { return 0.5 ; }

protected:

  virtual Bool_t processEmptyDataSets() const { return _extended ; }

  static RooArgSet _emptySet ; // Supports named argument constructor

  Bool_t _extended ;
  virtual Double_t evaluatePartition(Int_t firstEvent, Int_t lastEvent, Int_t stepSize) const ;
  Bool_t _weightSq ; // Apply weights squared?
  mutable Bool_t _first ; //!
  
  ClassDef(RooNLLVar,1) // Function representing (extended) -log(L) of p.d.f and dataset
};

#endif
