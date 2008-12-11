/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Class RooNLLVar implements a a -log(likelihood) calculation from a dataset
// and a PDF. The NLL is calculated as 
// <pre>
//  Sum[data] -log( pdf(x_data) )
// </pre>
// In extended mode, a (Nexpect - Nobserved*log(NExpected) term is added
// END_HTML
//

#include "RooFit.h"
#include "Riostream.h"

#include "RooNLLVar.h"
#include "RooAbsData.h"
#include "RooAbsPdf.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"

#include "RooRealVar.h"


ClassImp(RooNLLVar)
;

RooArgSet RooNLLVar::_emptySet ;


//_____________________________________________________________________________
RooNLLVar::RooNLLVar(const char *name, const char* title, RooAbsPdf& pdf, RooAbsData& data,
		     const RooCmdArg& arg1, const RooCmdArg& arg2,const RooCmdArg& arg3,
		     const RooCmdArg& arg4, const RooCmdArg& arg5,const RooCmdArg& arg6,
		     const RooCmdArg& arg7, const RooCmdArg& arg8,const RooCmdArg& arg9) :
  RooAbsOptTestStatistic(name,title,pdf,data,
			 *(const RooArgSet*)RooCmdConfig::decodeObjOnTheFly("RooNLLVar::RooNLLVar","ProjectedObservables",0,&_emptySet
									    ,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeStringOnTheFly("RooNLLVar::RooNLLVar","RangeWithName",0,"",arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeStringOnTheFly("RooNLLVar::RooNLLVar","AddCoefRange",0,"",arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","NumCPU",0,1,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 kFALSE,
			 RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","Verbose",0,1,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9),
			 RooCmdConfig::decodeIntOnTheFly("RooNLLVar::RooNLLVar","SplitRange",0,0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9))             
{
  // Construct likelihood from given p.d.f and (binned or unbinned dataset)
  //
  //  Extended()     -- Include extended term in calculation
  //  NumCPU()       -- Activate parallel processing feature
  //  Range()        -- Fit only selected region
  //  SumCoefRange() -- Set the range in which to interpret the coefficients of RooAddPdf components 
  //  SplitRange()   -- Fit range is split by index catory of simultaneous PDF
  //  ConditionalObservables() -- Define conditional observables 
  //  Verbose()      -- Verbose output of GOF framework classes

  RooCmdConfig pc("RooNLLVar::RooNLLVar") ;
  pc.allowUndefined() ;
  pc.defineInt("extended","Extended",0,kFALSE) ;

  pc.process(arg1) ;  pc.process(arg2) ;  pc.process(arg3) ;
  pc.process(arg4) ;  pc.process(arg5) ;  pc.process(arg6) ;
  pc.process(arg7) ;  pc.process(arg8) ;  pc.process(arg9) ;

  _extended = pc.getInt("extended") ;
}



//_____________________________________________________________________________
RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
		     Bool_t extended, const char* rangeName, const char* addCoefRangeName,
		     Int_t nCPU, Bool_t interleave, Bool_t verbose, Bool_t splitRange) : 
  RooAbsOptTestStatistic(name,title,pdf,data,RooArgSet(),rangeName,addCoefRangeName,nCPU,interleave,verbose,splitRange),
  _extended(extended)
{
  // Construct likelihood from given p.d.f and (binned or unbinned dataset)
  // For internal use.

}



//_____________________________________________________________________________
RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
		     const RooArgSet& projDeps, Bool_t extended, const char* rangeName,const char* addCoefRangeName, 
		     Int_t nCPU,Bool_t interleave,Bool_t verbose, Bool_t splitRange) : 
  RooAbsOptTestStatistic(name,title,pdf,data,projDeps,rangeName,addCoefRangeName,nCPU,interleave,verbose,splitRange),
  _extended(extended)
{
  // Construct likelihood from given p.d.f and (binned or unbinned dataset)
  // For internal use.  


}



//_____________________________________________________________________________
RooNLLVar::RooNLLVar(const RooNLLVar& other, const char* name) : 
  RooAbsOptTestStatistic(other,name),
  _extended(other._extended)
{
  // Copy constructor
}




//_____________________________________________________________________________
RooNLLVar::~RooNLLVar()
{
  // Destructor
}



//_____________________________________________________________________________
Double_t RooNLLVar::evaluatePartition(Int_t firstEvent, Int_t lastEvent, Int_t stepSize) const 
{
  // Calculate and return likelihood on subset of data from firstEvent to lastEvent
  // processed with a step size of 'stepSize'. If this an extended likelihood and
  // and the zero event is processed the extended term is added to the return
  // likelihood.

  Int_t i ;
  Double_t result(0) ;
  
  RooAbsPdf* pdfClone = (RooAbsPdf*) _funcClone ;

  Double_t sumWeight(0) ;
  for (i=firstEvent ; i<lastEvent ; i+=stepSize) {
    
    // get the data values for this event
    _dataClone->get(i);

    if (!_dataClone->valid()) {
      continue ;
    }

    if (_dataClone->weight()==0) continue ;

    // cout << "evaluating nll for event #" << i << " of " << lastEvent-firstEvent << endl ;
    Double_t term = _dataClone->weight() * pdfClone->getLogVal(_normSet);
    sumWeight += _dataClone->weight() ;

    //cout << "RooNLLVar term of event [" << i << "] is " << term << endl ; 

    result-= term;
  }
  
  // include the extended maximum likelihood term, if requested
  if(_extended && firstEvent==0) {
    result+= pdfClone->extendedTerm((Int_t)_dataClone->sumEntries(),_dataClone->get());
  }    

  // If part of simultaneous PDF normalize probability over 
  // number of simultaneous PDFs: -sum(log(p/n)) = -sum(log(p)) + N*log(n) 
  if (_simCount>1) {
    result += sumWeight*log(1.0*_simCount) ;
  }

  //cout << "RooNLLVar(first=" << firstEvent << ", last=" << lastEvent << ", step=" << stepSize << ") result = " << result << endl ;

  return result ;
}



