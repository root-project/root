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

#include "RooFit.h"

#include "RooParamBinning.h"
#include "RooParamBinning.h"
#include "RooMsgService.h"

#include "Riostream.h"


ClassImp(RooParamBinning)
;


RooParamBinning::RooParamBinning(const char* name) : 
  RooAbsBinning(name), _lp(0), _owner(0)
{  
  _array = 0 ;
}

RooParamBinning::RooParamBinning(RooAbsReal& xloIn, RooAbsReal& xhiIn, Int_t nBins, const char* name) :
  RooAbsBinning(name),
  _array(0), 
  _xlo(&xloIn),
  _xhi(&xhiIn),
  _nbins(nBins),
  _lp(0),
  _owner(0)
{
  //cout << "RooParamBinning::ctor(this = " << this << " xlo = " << &xloIn << " xhi = " << &xhiIn << " _lp = " << _lp << " owner = " << _owner << ")" << endl ;
}


RooParamBinning::~RooParamBinning() 
{
  //cout << "RooParamBinning::dtor(this = " << this << " xlo = " << &_xlo << " xhi = " << &_xhi << " _lp = " << _lp << " owner = " << _owner << ")" << endl ;

  if (_array) delete[] _array ;
  if (_lp) delete _lp ;
}


RooParamBinning::RooParamBinning(const RooParamBinning& other, const char* name) :
  RooAbsBinning(name), _owner(0)
{

  _array = 0 ;

  if (other._lp) {
    //cout << "RooParamBinning::cctor(this = " << this << " taking addresses from ListProxy" << endl ;
    _xlo = (RooAbsReal*) other._lp->at(0) ;
    _xhi = (RooAbsReal*) other._lp->at(1) ;

  } else {

    //cout << "RooParamBinning::cctor(this = " << this << " taking addresses from pointers " << endl ;

    _xlo   = other._xlo ;
    _xhi   = other._xhi ;
  }

  _nbins = other._nbins ;
  _lp = 0 ;

  //cout << "RooParamBinning::cctor(this = " << this << " xlo = " << &_xlo << " xhi = " << &_xhi << " _lp = " << _lp << " owner = " << _owner << ")" << endl ;
}


void RooParamBinning::insertHook(RooAbsRealLValue& owner) const  
{
  _owner = &owner ;


  if (_lp) {
    //cout << "RooParamBinning::insertHook(this = " << this << " deleting previous _lp  = " << _lp << endl ;
    _xlo = xlo() ;
    _xhi = xhi() ;
    delete _lp ;
  }

  _lp = new RooListProxy("lp","lp",&owner,kFALSE,kTRUE) ;
  _lp->add(*_xlo) ;
  _lp->add(*_xhi) ;
  _xlo = 0 ;
  _xhi = 0 ;

  //cout << "RooParamBinning::insertHook(this = " << this << " xlo = " << &_xlo << " xhi = " << &_xhi << " _lp = " << _lp << " owner = " << _owner << ")" << endl ;

}

void RooParamBinning::removeHook(RooAbsRealLValue& /*owner*/) const  
{
  _owner = 0 ;
  
  //cout << "RooParamBinning::removeHook(this = " << this << " xlo = " << &_xlo << " xhi = " << &_xhi << " _lp = " << _lp << " owner = " << _owner << ")" << endl ;


  if (_lp) {
    //cout << "RooParamBinning::insertHook(this = " << this << " deleting previous _lp  = " << _lp << endl ;
    _xlo = xlo() ;
    _xhi = xhi() ;
    delete _lp ;
    _lp = 0 ;
  }
}


void RooParamBinning::setRange(Double_t newxlo, Double_t newxhi) 
{
  if (newxlo>newxhi) {
    coutE(InputArguments) << "RooParamBinning::setRange: ERROR low bound > high bound" << endl ;
    return ;
  }

  RooAbsRealLValue* xlolv = dynamic_cast<RooAbsRealLValue*>(xlo()) ;
  if (xlolv) {
    xlolv->setVal(newxlo) ;
  } else {
    coutW(InputArguments) << "RooParamBinning::setRange: WARNING lower bound not represented by lvalue, cannot set lower bound value through setRange()" << endl ;
  }

  RooAbsRealLValue* xhilv = dynamic_cast<RooAbsRealLValue*>(xhi()) ;
  if (xhilv) {
    xhilv->setVal(newxhi) ;
  } else {
    coutW(InputArguments) << "RooParamBinning::setRange: WARNING upper bound not represented by lvalue, cannot set upper bound value through setRange()" << endl ;
  }

}



Int_t RooParamBinning::binNumber(Double_t x) const  
{
  // Return the fit bin index for the current value
  if (x >= xhi()->getVal()) return _nbins-1 ;
  if (x < xlo()->getVal()) return 0 ;

  return Int_t((x - xlo()->getVal())/averageBinWidth()) ;
}


Double_t RooParamBinning::binCenter(Int_t i) const 
{
  // Return the central value of the 'i'-th fit bin
  if (i<0 || i>=_nbins) {
    coutE(InputArguments) << "RooParamBinning::binCenter ERROR: bin index " << i 
			  << " is out of range (0," << _nbins-1 << ")" << endl ;
    return 0 ;
  }

  return xlo()->getVal() + (i + 0.5)*averageBinWidth() ;  
}




Double_t RooParamBinning::binWidth(Int_t /*bin*/) const 
{
  return (xhi()->getVal()-xlo()->getVal())/_nbins ;
}



Double_t RooParamBinning::binLow(Int_t i) const 
{
  // Return the low edge of the 'i'-th fit bin
  if (i<0 || i>=_nbins) {
    coutE(InputArguments) << "RooParamBinning::binLow ERROR: bin index " << i 
			  << " is out of range (0," << _nbins-1 << ")" << endl ;
    return 0 ;
  }

  return xlo()->getVal() + i*binWidth(i) ;
}


Double_t RooParamBinning::binHigh(Int_t i) const 
{
  // Return the high edge of the 'i'-th fit bin
  if (i<0 || i>=_nbins) {
    coutE(InputArguments) << "RooParamBinning::fitBinHigh ERROR: bin index " << i 
			  << " is out of range (0," << _nbins-1 << ")" << endl ;
    return 0 ;
  }

  return xlo()->getVal() + (i + 1)*binWidth(i) ;
}



Double_t* RooParamBinning::array() const 
{
  if (_array) delete[] _array ;
  _array = new Double_t[_nbins+1] ;

  Int_t i ;
  for (i=0 ; i<=_nbins ; i++) {
    _array[i] = xlo()->getVal() + i*binWidth(i) ;
  }
  return _array ;
}


void RooParamBinning::printMultiline(ostream &os, Int_t /*content*/, Bool_t /*verbose*/, TString indent) const
{
  os << indent << "_xlo = " << _xlo << endl ;
  os << indent << "_xhi = " << _xhi << endl ;
  if (_lp) {
    os << indent << "xlo() = " << xlo() << endl ;
    os << indent << "xhi() = " << xhi() << endl ;
  }  
}


