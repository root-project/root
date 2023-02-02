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
\file RooUniformBinning.cxx
\class RooUniformBinning
\ingroup Roofitcore

RooUniformBinning is an implementation of RooAbsBinning that provides
a uniform binning in 'n' bins between the range end points. A RooUniformBinning
is 'elastic': if the range changes the binning will change accordingly, unlike
e.g. the binning of class RooBinning.
**/

#include "RooUniformBinning.h"
#include "RooMsgService.h"

#include "Riostream.h"


using namespace std;

ClassImp(RooUniformBinning);
;



////////////////////////////////////////////////////////////////////////////////
/// Default Constructor
/// coverity[UNINIT_CTOR]

RooUniformBinning::RooUniformBinning(const char* name) :
  RooAbsBinning(name)
{
  _array = 0 ;
}


////////////////////////////////////////////////////////////////////////////////
/// Construct range [xlo,xhi] with 'nBins' bins

RooUniformBinning::RooUniformBinning(double xlo, double xhi, Int_t nBins, const char* name) :
  RooAbsBinning(name),
  _array(0),
  _nbins(nBins)
{
  setRange(xlo,xhi) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooUniformBinning::~RooUniformBinning()
{
  if (_array) delete[] _array ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooUniformBinning::RooUniformBinning(const RooUniformBinning& other, const char* name) :
  RooAbsBinning(name)
{
  _array = 0 ;
  _xlo   = other._xlo ;
  _xhi   = other._xhi ;
  _nbins = other._nbins ;
  _binw  = other._binw ;
}



////////////////////////////////////////////////////////////////////////////////
/// Change range to [xlo,xhi]. A changes in range automatically
/// adjusts the binning as well to nBins bins in the new range

void RooUniformBinning::setRange(double xlo, double xhi)
{
  if (xlo>xhi) {
    coutE(InputArguments) << "RooUniformBinning::setRange: ERROR low bound > high bound" << endl ;
    return ;
  }

  _xlo = xlo ;
  _xhi = xhi ;
  _binw = (xhi-xlo)/_nbins ;

  // Delete any out-of-date boundary arrays at this point
  if (_array) {
    delete[] _array ;
    _array = 0 ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return the index of the bin that encloses 'x'

void RooUniformBinning::binNumbers(double const * x, int * bins, std::size_t n, int coef) const
{
  const double oneOverW = 1./_binw;

  for(std::size_t i = 0; i < n; ++i) {
    bins[i] += coef * (x[i] >= _xhi ? _nbins - 1 : std::max(0, int((x[i] - _xlo)*oneOverW)));
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return the central value of the 'i'-th fit bin

double RooUniformBinning::binCenter(Int_t i) const
{
  if (i<0 || i>=_nbins) {
    coutE(InputArguments) << "RooUniformBinning::binCenter ERROR: bin index " << i
           << " is out of range (0," << _nbins-1 << ")" << endl ;
    return 0 ;
  }

  return _xlo + (i + 0.5) * _binw;
}




////////////////////////////////////////////////////////////////////////////////
/// Return the bin width (same for all bins)

double RooUniformBinning::binWidth(Int_t /*bin*/) const
{
  return _binw ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the low edge of the 'i'-th fit bin

double RooUniformBinning::binLow(Int_t i) const
{
  if (i<0 || i>=_nbins) {
    coutE(InputArguments) << "RooUniformBinning::binLow ERROR: bin index " << i
           << " is out of range (0," << _nbins-1 << ")" << endl ;
    return 0 ;
  }

  return _xlo + i*_binw ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the high edge of the 'i'-th fit bin

double RooUniformBinning::binHigh(Int_t i) const
{
  if (i<0 || i>=_nbins) {
    coutE(InputArguments) << "RooUniformBinning::fitBinHigh ERROR: bin index " << i
           << " is out of range (0," << _nbins-1 << ")" << endl ;
    return 0 ;
  }

  return _xlo + (i + 1)*_binw ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return an array of doubles with the bin boundaries

double* RooUniformBinning::array() const
{
  if (_array) delete[] _array ;
  _array = new double[_nbins+1] ;

  Int_t i ;
  for (i=0 ; i<=_nbins ; i++) {
    _array[i] = _xlo + i*_binw ;
  }
  return _array ;
}


