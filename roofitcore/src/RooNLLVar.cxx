/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNLLVar.cc,v 1.19 2005/06/20 15:44:55 wverkerke Exp $
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

// -- CLASS DESCRIPTION [PDF] --
// Class RooNLLVar implements a a -log(likelihood) calculation from a dataset
// and a PDF. The NLL is calculated as 
//
//  Sum[data] -log( pdf(x_data) )
//
// In extended mode, a (Nexpect - Nobserved*log(NExpected) term is added

#include "RooFit.h"

#include "RooNLLVar.h"
#include "RooNLLVar.h"
#include "RooAbsData.h"
#include "RooAbsPdf.h"
#include "RooCmdConfig.h"

ClassImp(RooNLLVar)
;

RooArgSet RooNLLVar::_emptySet ;

RooNLLVar::RooNLLVar(const char *name, const char* title, RooAbsPdf& pdf, RooAbsData& data,
		     const RooCmdArg& arg1, const RooCmdArg& arg2,const RooCmdArg& arg3,
		     const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6,
		     const RooCmdArg& arg7, const RooCmdArg& arg8,const RooCmdArg& arg9) :
  RooAbsOptGoodnessOfFit(name,title,pdf,data,
			 *(const RooArgSet*)RooCmdConfig::decodeObjOnTheFly("RooNLLVar::RooNLLVar","ProjectedObservables",0,&_emptySet
									    ,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeStringOnTheFly("RooNLLVar::RooNLLVar","RangeWithName",0,"",arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","NumCPU",0,1,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","Verbose",0,1,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","SplitRange",0,0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9))             
{
  // RooNLLVar constructor. Optional arguments taken
  //
  //  Extended()   -- Include extended term in calculation
  //  NumCPU()     -- Activate parallel processing feature
  //  Range()      -- Fit only selected region
  //  SplitRange() -- Fit range is split by index catory of simultaneous PDF
  //  ConditionalObservables() -- Define conditional observables 
  //  Verbose()    -- Verbose output of GOF framework classes

  RooCmdConfig pc("RooNLLVar::RooNLLVar") ;
  pc.defineInt("extended","Extended",0,kFALSE) ;

  pc.process(arg1) ;  pc.process(arg2) ;  pc.process(arg3) ;
  pc.process(arg4) ;  pc.process(arg5) ;  pc.process(arg6) ;
  pc.process(arg7) ;  pc.process(arg8) ;  pc.process(arg9) ;

  _extended = pc.getInt("extended") ;
}



RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
		     Bool_t extended, const char* rangeName, Int_t nCPU, Bool_t verbose, Bool_t splitRange) : 
  RooAbsOptGoodnessOfFit(name,title,pdf,data,RooArgSet(),rangeName,nCPU,verbose,splitRange),
  _extended(extended)
{
  
}


RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
		     const RooArgSet& projDeps, Bool_t extended, const char* rangeName,Int_t nCPU,Bool_t verbose, Bool_t splitRange) : 
  RooAbsOptGoodnessOfFit(name,title,pdf,data,projDeps,rangeName,nCPU,verbose,splitRange),
  _extended(extended)
{
  
}


RooNLLVar::RooNLLVar(const RooNLLVar& other, const char* name) : 
  RooAbsOptGoodnessOfFit(other,name),
  _extended(other._extended)
{
}


RooNLLVar::~RooNLLVar()
{
}


Double_t RooNLLVar::evaluatePartition(Int_t firstEvent, Int_t lastEvent) const 
{
  Int_t i ;
  Double_t result(0) ;
  
  Double_t sumWeight(0) ;
  for (i=firstEvent ; i<lastEvent ; i++) {
    
    // get the data values for this event
    _dataClone->get(i);
    if (_dataClone->weight()==0) continue ;

    //cout << "RooNLLVar(" << GetName() << ") wgt[" << i << "] = " << _dataClone->weight() << endl ;

    Double_t term = _dataClone->weight() * _pdfClone->getLogVal(_normSet);
    sumWeight += _dataClone->weight() ;

    // If any event evaluates with zero probability, abort calculation
    if(term == 0) {
      cout << "RooNLLVar::evaluatePartition(" << GetName() 
	   << "): WARNING: event " << i << " has zero or negative probability" << endl ;
      return 0 ;
    }

    result-= term;
  }
  
  // include the extended maximum likelihood term, if requested
  if(_extended && firstEvent==0) {
    result+= _pdfClone->extendedTerm((Int_t)_dataClone->sumEntries(),_dataClone->get());
  }    

  // If part of simultaneous PDF normalize probability over 
  // number of simultaneous PDFs: -sum(log(p/n)) = -sum(log(p)) + N*log(n) 
  if (_simCount>1) {
    result += sumWeight*log(1.0*_simCount) ;
  }

  return result ;
}



