/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
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

// -- CLASS DESCRIPTION [PDF] --
// Class RooNLLVar implements a a -log(likelihood) calculation from a dataset
// and a PDF. The NLL is calculated as 
//
//  Sum[data] -log( pdf(x_data) )
//
// In extended mode, a (Nexpect - Nobserved*log(NExpected) term is added

#include "RooFitCore/RooNLLVar.hh"
#include "RooFitCore/RooAbsData.hh"
#include "RooFitCore/RooAbsPdf.hh"

ClassImp(RooNLLVar)
;

RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
		     Bool_t extended, Int_t nCPU) : 
  RooAbsOptGoodnessOfFit(name,title,pdf,data,RooArgSet(),nCPU),
  _extended(extended)
{
  
}


RooNLLVar::RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
		     const RooArgSet& projDeps, Bool_t extended, Int_t nCPU) : 
  RooAbsOptGoodnessOfFit(name,title,pdf,data,projDeps,nCPU),
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
  
  for (i=firstEvent ; i<lastEvent ; i++) {
    
    // get the data values for this event
    _dataClone->get(i);
    Double_t term = _dataClone->weight() * _pdfClone->getLogVal(_normSet);

    // If any event evaluates with zero probability, abort calculation
    if(term == 0 && (_dataClone->weight()!=0.)) {
      cout << "RooNLLVar::evaluatePartition(" << GetName() 
	   << "): WARNING: event " << i << " has zero or negative probability" << endl ;
      return 0 ;
    }

    result-= term;
  }
  
  // include the extended maximum likelihood term, if requested
  if(_extended && firstEvent==0) {
    result+= _pdfClone->extendedTerm(_dataClone->numEntries(kTRUE));
  }    

  // If part of simultaneous PDF normalize probability over 
  // number of simultaneous PDFs: -sum(log(p/n)) = -sum(log(p)) + N*log(n) 
  if (_simCount>1) {
    result += (lastEvent-firstEvent)*log(_simCount) ;
  }

  return result ;
}



