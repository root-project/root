/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooRealFixedBinIter.hh"
#include "RooFitCore/RooAbsReal.hh"

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
  _curCenter = ((RooAbsReal*)_arg)->getPlotMin() ;
}

RooAbsArg* RooRealFixedBinIter::next() 
{
  // Check for upper limit
  if (_curBin>=((RooAbsReal*)_arg)->getPlotBins()) return 0 ;

  // Increment bin counter and central value
  _curBin++ ;
  _curCenter += ((RooAbsReal*)_arg)->getPlotBinWidth() ;

  // Return RooRealVar 
  return _arg ;
}
  
Double_t RooRealFixedBinIter::currentCenter() const 
{
  return _curCenter ;
}


Double_t RooRealFixedBinIter::currentLow() const  
{
  return ((RooAbsReal*)_arg)->plotBinLow(_curBin) ;
}


Double_t RooRealFixedBinIter::currentHigh() const 
{
  return ((RooAbsReal*)_arg)->plotBinHigh(_curBin) ;
}

