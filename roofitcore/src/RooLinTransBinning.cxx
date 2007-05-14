/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooLinTransBinning.cxx,v 1.13 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [AUX] --
// RooLinTransBinning is a special binning implementation for RooLinearVar
// that transforms the binning of the RooLinearVar input variable in the same
// way that RooLinearVar does


#include "RooFit.h"

#include "RooLinTransBinning.h"
#include "RooLinTransBinning.h"

ClassImp(RooLinTransBinning) 
;


RooLinTransBinning::RooLinTransBinning(const RooAbsBinning& input, Double_t slope, Double_t offset, const char* name) :
  RooAbsBinning(name),
  _array(0) 
{
  // Constructor
  updateInput(input,slope,offset) ;
}



RooLinTransBinning::RooLinTransBinning(const RooLinTransBinning& other, const char* name) :
  RooAbsBinning(name),
  _array(0)
{
  // Copy constructor
  _input = other._input ;
  _slope = other._slope ;
  _offset = other._offset ;    
}


RooLinTransBinning::~RooLinTransBinning() 
{
  // Destructor 
  if (_array) delete[] _array ;
}


void RooLinTransBinning::setRange(Double_t /*xlo*/, Double_t /*xhi*/) 
{
  // Change limits
}

Double_t* RooLinTransBinning::array() const 
{
  Int_t n = numBoundaries() ;
  // Return array with boundary values
  if (_array) delete[] _array ;
  _array = new Double_t[n] ;

  Double_t* inputArray = _input->array() ;

  Int_t i ;
  if (_slope>0) {
    for (i=0 ; i<n ; i++) {
      _array[i] = trans(inputArray[i]) ;
    }
  } else {
    for (i=0 ; i<n ; i++) {
      _array[i] = trans(inputArray[n-i-1]) ;
    }
  }
  return _array ;

}


void RooLinTransBinning::updateInput(const RooAbsBinning& input, Double_t slope, Double_t offset)
{
  _input = (RooAbsBinning*) &input ;
  _slope = slope ;
  _offset = offset ;
}

