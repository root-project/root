/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooUniformBinning.cc,v 1.3 2002/03/28 00:14:38 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, UC Irvine, davidk@slac.stanford.edu
 * History:
 *   01-Mar-2002 WV Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

#include "RooFitCore/RooUniformBinning.hh"

ClassImp(RooUniformBinning)
;


RooUniformBinning::RooUniformBinning() 
{  
  _array = 0 ;
}

RooUniformBinning::RooUniformBinning(Double_t xlo, Double_t xhi, Int_t nBins) : _nbins(nBins), _array(0)
{
  setRange(xlo,xhi) ;
}


RooUniformBinning::~RooUniformBinning() 
{
  if (_array) delete[] _array ;
}


RooUniformBinning::RooUniformBinning(const RooUniformBinning& other) 
{
  _array = 0 ;
  _xlo   = other._xlo ;
  _xhi   = other._xhi ;
  _nbins = other._nbins ;
  _binw  = other._binw ;  
}


void RooUniformBinning::setRange(Double_t xlo, Double_t xhi) 
{
  if (xlo>xhi) {
    cout << "RooUniformBinning::setRange: ERROR low bound > high bound" << endl ;
    return ;
  }
  
  _xlo = xlo ;
  _xhi = xhi ;
  _binw = (xhi-xlo)/_nbins ;
}



Int_t RooUniformBinning::binNumber(Double_t x) const  
{
  // Return the fit bin index for the current value
  if (x >= _xhi) return _nbins-1 ;
  if (x < _xlo) return 0 ;

  return Int_t((x - _xlo)/_binw) ;
}


Double_t RooUniformBinning::binCenter(Int_t i) const 
{
  // Return the central value of the 'i'-th fit bin
  if (i<0 || i>=_nbins) {
    cout << "RooUniformBinning::binCenter ERROR: bin index " << i 
	 << " is out of range (0," << _nbins-1 << ")" << endl ;
    return 0 ;
  }

  return _xlo + (i + 0.5)*averageBinWidth() ;  
}




Double_t RooUniformBinning::binWidth(Int_t bin) const 
{
  return _binw ;
}



Double_t RooUniformBinning::binLow(Int_t i) const 
{
  // Return the low edge of the 'i'-th fit bin
  if (i<0 || i>=_nbins) {
    cout << "RooUniformBinning::binLow ERROR: bin index " << i 
	 << " is out of range (0," << _nbins-1 << ")" << endl ;
    return 0 ;
  }

  return _xlo + i*_binw ;
}


Double_t RooUniformBinning::binHigh(Int_t i) const 
{
  // Return the high edge of the 'i'-th fit bin
  if (i<0 || i>=_nbins) {
    cout << "RooUniformBinning::fitBinHigh ERROR: bin index " << i 
	 << " is out of range (0," << _nbins-1 << ")" << endl ;
    return 0 ;
  }

  return _xlo + (i + 1)*_binw ;
}



Double_t* RooUniformBinning::array() const 
{
  if (_array) delete[] _array ;
  _array = new Double_t[_nbins+1] ;

  Int_t i ;
  for (i=0 ; i<=_nbins ; i++) {
    _array[i] = _xlo + i*_binw ;
  }
  return _array ;
}


void RooUniformBinning::printToStream(ostream &os, PrintOption opt, TString indent) const
{
  if (opt==Standard) {
    os << "B(" << _nbins << ")" << endl ;
  }
}

