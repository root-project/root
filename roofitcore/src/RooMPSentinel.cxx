/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --
// RooMPSentinel is a singleton class that keeps track of all
// parellel execution processes for goodness-of-fit calculations.
// The primary task of RooMPSentinel is to terminate all server processes
// when the main ROOT process is exiting.


#include "RooFitCore/RooMPSentinel.hh"
#include "RooFitCore/RooRealMPFE.hh"

ClassImp(RooMPSentinel)
  ;

RooMPSentinel::RooMPSentinel() 
{
}


RooMPSentinel::~RooMPSentinel() 
{
  TIterator *iter = _mpfeSet.createIterator() ;
  RooRealMPFE* mpfe ;
  while(mpfe=(RooRealMPFE*)iter->Next()) {
    mpfe->standby() ;
  }
  delete iter ;
}
 

void RooMPSentinel::add(RooRealMPFE& mpfe) 
{
  _mpfeSet.add(mpfe,kTRUE) ;
}


void RooMPSentinel::remove(RooRealMPFE& mpfe) 
{
  _mpfeSet.remove(mpfe,kTRUE) ;
}
