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

/**
\file RooLinTransBinning.cxx
\class RooLinTransBinning
\ingroup Roofitcore

RooLinTransBinning is a special binning implementation for RooLinearVar
that transforms the binning of the RooLinearVar input variable in the same
way that RooLinearVar does
**/


#include "RooFit.h"

#include "RooLinTransBinning.h"
#include "RooLinTransBinning.h"

using namespace std;

ClassImp(RooLinTransBinning) 
;



////////////////////////////////////////////////////////////////////////////////
/// Constructor with a given input binning and the slope and offset to be applied to
/// construct the linear transformation

RooLinTransBinning::RooLinTransBinning(const RooAbsBinning& input, Double_t slope, Double_t offset, const char* name) :
  RooAbsBinning(name),
  _array(0) 
{
  updateInput(input,slope,offset) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooLinTransBinning::RooLinTransBinning(const RooLinTransBinning& other, const char* name) :
  RooAbsBinning(name),
  _array(0)
{
  _input = other._input ;
  _slope = other._slope ;
  _offset = other._offset ;    
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor 

RooLinTransBinning::~RooLinTransBinning() 
{
  if (_array) delete[] _array ;
}



////////////////////////////////////////////////////////////////////////////////

void RooLinTransBinning::setRange(Double_t /*xlo*/, Double_t /*xhi*/) 
{
  // Change limits -- not implemented
}


////////////////////////////////////////////////////////////////////////////////
/// Return array of bin boundaries

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



////////////////////////////////////////////////////////////////////////////////
/// Update the slope and offset parameters and the pointer to the input binning

void RooLinTransBinning::updateInput(const RooAbsBinning& input, Double_t slope, Double_t offset)
{
  _input = (RooAbsBinning*) &input ;
  _slope = slope ;
  _offset = offset ;
}

