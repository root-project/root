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
\file RooParamBinning.cxx
\class RooParamBinning
\ingroup Roofitcore

Implementation of RooAbsBinning that constructs
a binning with a range definition that depends on external RooAbsReal objects.
The external RooAbsReal definitions are explicitly allowed to depend on other
observables and parameters, and make it possible to define non-rectangular
range definitions in RooFit. Objects of class RooParamBinning are made
by the RooRealVar::setRange() that takes RooAbsReal references as arguments
**/

#include "RooParamBinning.h"
#include "RooMsgService.h"

#include "Riostream.h"


using std::endl, std::ostream;



////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooParamBinning::RooParamBinning(const char* name) :
  RooAbsBinning(name)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Construct binning with 'nBins' bins and with a range
/// parameterized by external RooAbsReals xloIn and xhiIn.

RooParamBinning::RooParamBinning(RooAbsReal& xloIn, RooAbsReal& xhiIn, Int_t nBins, const char* name) :
  RooAbsBinning(name),
  _xlo(&xloIn),
  _xhi(&xhiIn),
  _nbins(nBins)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooParamBinning::~RooParamBinning()
{
  if (_array) delete[] _array ;
  if (_lp) delete _lp ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor
///   std::cout << "RooParamBinning::cctor(" << this << ") orig = " << &other << std::endl ;

RooParamBinning::RooParamBinning(const RooParamBinning &other, const char *name) : RooAbsBinning(name)
{

  if (other._lp) {
//     std::cout << "RooParamBinning::cctor(this = " << this << ") taking addresses from orig  ListProxy" << std::endl ;
    _xlo = static_cast<RooAbsReal*>(other._lp->at(0)) ;
    _xhi = static_cast<RooAbsReal*>(other._lp->at(1)) ;

  } else {

//     std::cout << "RooParamBinning::cctor(this = " << this << ") taking addresses from orig pointers " << other._xlo << " " << other._xhi << std::endl ;

    _xlo   = other._xlo ;
    _xhi   = other._xhi ;
  }

  _nbins = other._nbins ;
  _lp = nullptr ;

  //cout << "RooParamBinning::cctor(this = " << this << " xlo = " << &_xlo << " xhi = " << &_xhi << " _lp = " << _lp << " owner = " << _owner << ")" << std::endl ;
}



////////////////////////////////////////////////////////////////////////////////
/// Hook function called by RooAbsRealLValue when this binning
/// is inserted as binning for into given owner. Create
/// list proxy registered with owner that will track and implement
/// server directs to external RooAbsReals of this binning

void RooParamBinning::insertHook(RooAbsRealLValue& owner) const
{
  _owner = &owner ;

  // If list proxy already exists update pointers from proxy
//   std::cout << "RooParamBinning::insertHook(" << this << "," << GetName() << ") _lp at beginning = " << _lp << std::endl ;
  if (_lp) {
//     std::cout << "updating raw pointers from list proxy contents" << std::endl ;
    _xlo = xlo() ;
    _xhi = xhi() ;
    delete _lp ;
  }
//   std::cout << "_xlo = " << _xlo << " _xhi = " << _xhi << std::endl ;

  // If list proxy does not exist, create it now
  _lp = new RooListProxy(Form("range::%s",GetName()),"lp",&owner,false,true) ;
  _lp->add(*_xlo) ;
  _lp->add(*_xhi) ;
  _xlo = nullptr ;
  _xhi = nullptr ;


}


////////////////////////////////////////////////////////////////////////////////
/// Hook function called by RooAbsRealLValue when this binning
/// is removed as binning for into given owner. Delete list
/// proxy that was inserted in owner

void RooParamBinning::removeHook(RooAbsRealLValue& /*owner*/) const
{
  _owner = nullptr ;

  // Remove list proxy from owner
  if (_lp) {
    _xlo = xlo() ;
    _xhi = xhi() ;
    delete _lp ;
    _lp = nullptr ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Adjust range by adjusting values of external RooAbsReal values
/// Only functional when external representations are lvalues

void RooParamBinning::setRange(double newxlo, double newxhi)
{
  if (newxlo>newxhi) {
    coutE(InputArguments) << "RooParamBinning::setRange: ERROR low bound > high bound" << std::endl ;
    return ;
  }

  RooAbsRealLValue* xlolv = dynamic_cast<RooAbsRealLValue*>(xlo()) ;
  if (xlolv) {
    xlolv->setVal(newxlo) ;
  } else {
    coutW(InputArguments) << "RooParamBinning::setRange: WARNING lower bound not represented by lvalue, cannot set lower bound value through setRange()" << std::endl ;
  }

  RooAbsRealLValue* xhilv = dynamic_cast<RooAbsRealLValue*>(xhi()) ;
  if (xhilv) {
    xhilv->setVal(newxhi) ;
  } else {
    coutW(InputArguments) << "RooParamBinning::setRange: WARNING upper bound not represented by lvalue, cannot set upper bound value through setRange()" << std::endl ;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Return the fit bin index for the current value

void RooParamBinning::binNumbers(double const * x, int * bins, std::size_t n, int coef) const
{
  const double xloVal = xlo()->getVal();
  const double xhiVal = xhi()->getVal();
  const double oneOverW = 1./averageBinWidth();

  for(std::size_t i = 0; i < n; ++i) {
    bins[i] += coef * (x[i] >= xhiVal ? _nbins - 1 : std::max(0, int((x[i] - xloVal)*oneOverW)));
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return the central value of the 'i'-th fit bin

double RooParamBinning::binCenter(Int_t i) const
{
  if (i<0 || i>=_nbins) {
    coutE(InputArguments) << "RooParamBinning::binCenter ERROR: bin index " << i
           << " is out of range (0," << _nbins-1 << ")" << std::endl ;
    return 0 ;
  }

  return xlo()->getVal() + (i + 0.5)*averageBinWidth() ;
}




////////////////////////////////////////////////////////////////////////////////
/// Return average bin width

double RooParamBinning::binWidth(Int_t /*bin*/) const
{
  return (xhi()->getVal()-xlo()->getVal())/_nbins ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the low edge of the 'i'-th fit bin

double RooParamBinning::binLow(Int_t i) const
{
  if (i<0 || i>=_nbins) {
    coutE(InputArguments) << "RooParamBinning::binLow ERROR: bin index " << i
           << " is out of range (0," << _nbins-1 << ")" << std::endl ;
    return 0 ;
  }

  return xlo()->getVal() + i*binWidth(i) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the high edge of the 'i'-th fit bin

double RooParamBinning::binHigh(Int_t i) const
{
  if (i<0 || i>=_nbins) {
    coutE(InputArguments) << "RooParamBinning::fitBinHigh ERROR: bin index " << i
           << " is out of range (0," << _nbins-1 << ")" << std::endl ;
    return 0 ;
  }

  return xlo()->getVal() + (i + 1)*binWidth(i) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return array of bin boundaries

double* RooParamBinning::array() const
{
  if (_array) delete[] _array ;
  _array = new double[_nbins+1] ;

  Int_t i ;
  for (i=0 ; i<=_nbins ; i++) {
    _array[i] = xlo()->getVal() + i*binWidth(i) ;
  }
  return _array ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print details of binning

void RooParamBinning::printMultiline(ostream &os, Int_t /*content*/, bool /*verbose*/, TString indent) const
{
  os << indent << "_xlo = " << _xlo << std::endl ;
  os << indent << "_xhi = " << _xhi << std::endl ;
  if (_lp) {
    os << indent << "xlo() = " << xlo() << std::endl ;
    os << indent << "xhi() = " << xhi() << std::endl ;
  }
  if (xlo()) {
    xlo()->Print("t") ;
  }
  if (xhi()) {
    xhi()->Print("t") ;
  }
}
