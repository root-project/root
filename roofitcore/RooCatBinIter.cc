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

#include "RooFitCore/RooCatBinIter.hh"
#include "RooFitCore/RooAbsCategory.hh"

ClassImp(RooCatBinIter) 
;


RooCatBinIter::RooCatBinIter(const RooAbsArg& arg) :
  RooAbsBinIter(arg) 
{
}

RooCatBinIter::RooCatBinIter(const RooCatBinIter& other) :
  RooAbsBinIter(other)
{
}


RooCatBinIter::~RooCatBinIter() 
{
}
  

void RooCatBinIter::reset()
{
  _curBin = 0 ;
}

RooAbsArg* RooCatBinIter::next() 
{
  // Check for upper limitg
  if (_curBin>=((RooAbsCategory*)_arg)->numTypes()) return 0 ;

  // Increment bin counter and central value
  _curBin++ ;

  // Return RooRealVar 
  return _arg ;
}
  
Double_t RooCatBinIter::currentCenter() const 
{
  return _curBin ;
}


Double_t RooCatBinIter::currentLow() const  
{
  return _curBin-0.5 ;
}


Double_t RooCatBinIter::currentHigh() const 
{
  return _curBin+0.5 ;
}

