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

#include "RooFitCore/RooLinTransBinning.hh"

ClassImp(RooLinTransBinning) 
;


RooLinTransBinning::RooLinTransBinning(const RooAbsBinning& input, Double_t slope, Double_t offset) : _array(0) 
{
  // Constructor
  updateInput(input,slope,offset) ;
}



RooLinTransBinning::RooLinTransBinning(const RooLinTransBinning& other) : _array(0)
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


void RooLinTransBinning::setRange(Double_t xlo, Double_t xhi) 
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


void RooLinTransBinning::updateInput(const RooAbsBinning& input, Double_t slope=1.0, Double_t offset=0.0)
{
  _input = (RooAbsBinning*) &input ;
  _slope = slope ;
  _offset = offset ;
}

