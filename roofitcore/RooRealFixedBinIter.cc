/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealFixedBinIter.cc,v 1.2 2001/09/27 18:22:30 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [AUX] --

#include "RooFitCore/RooRealFixedBinIter.hh"
#include "RooFitCore/RooAbsRealLValue.hh"

ClassImp(RooRealFixedBinIter) 
;


RooRealFixedBinIter::RooRealFixedBinIter(const RooAbsArg& arg) :
  RooAbsBinIter(arg) 
{
}

RooRealFixedBinIter::RooRealFixedBinIter(const RooRealFixedBinIter& other) :
  RooAbsBinIter(other)
{
}


RooRealFixedBinIter::~RooRealFixedBinIter() 
{
}
  

void RooRealFixedBinIter::reset()
{
  _curBin = 0 ;
  _curCenter = ((RooAbsRealLValue*)_arg)->getFitMin() ;
}

RooAbsArg* RooRealFixedBinIter::next() 
{
  // Check for upper limit
  if (_curBin>=((RooAbsRealLValue*)_arg)->getFitBins()) return 0 ;

  // Increment bin counter and central value
  _curBin++ ;
  _curCenter += ((RooAbsRealLValue*)_arg)->getFitBinWidth() ;

  // Return RooRealVar 
  return _arg ;
}
  
Double_t RooRealFixedBinIter::currentCenter() const 
{
  return _curCenter ;
}


Double_t RooRealFixedBinIter::currentLow() const  
{
  return ((RooAbsRealLValue*)_arg)->fitBinLow(_curBin) ;
}


Double_t RooRealFixedBinIter::currentHigh() const 
{
  return ((RooAbsRealLValue*)_arg)->fitBinHigh(_curBin) ;
}

