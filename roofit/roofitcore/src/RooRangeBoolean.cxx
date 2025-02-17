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
\file RooRangeBoolean.cxx
\class RooRangeBoolean
\ingroup Roofitcore

Returns `1.0` if variable is within given a range and `0.0` otherwise.
**/

#include "Riostream.h"
#include <cmath>

#include "RooRangeBoolean.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"


////////////////////////////////////////////////////////////////////////////////

RooRangeBoolean::RooRangeBoolean(const char* name, const char* title, RooAbsRealLValue& x, const char* rangeName) :
  RooAbsReal(name, title),
  _x("x", "Dependent", this, x),
  _rangeName(rangeName)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooRangeBoolean::RooRangeBoolean(const RooRangeBoolean& other, const char* name) :
  RooAbsReal(other, name),
  _x("x", this, other._x),
  _rangeName(other._rangeName)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Return 1 if x is in range, zero otherwise.

double RooRangeBoolean::evaluate() const
{
  double xmin = (static_cast<RooAbsRealLValue const&>(_x.arg())).getMin(_rangeName.Data()) ;
  double xmax = (static_cast<RooAbsRealLValue const&>(_x.arg())).getMax(_rangeName.Data()) ;

  double ret = (_x >= xmin && _x < xmax) ? 1.0 : 0.0 ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////

std::list<double>* RooRangeBoolean::plotSamplingHint(RooAbsRealLValue& obs, double /*xlo*/, double /*xhi*/) const
{
  if (std::string(obs.GetName())!=_x.arg().GetName()) {
    return nullptr ;
  }

  std::list<double>* hint = new std::list<double> ;
  hint->push_back((static_cast<RooAbsRealLValue const&>(_x.arg())).getMin(_rangeName.Data())-1e-6) ;
  hint->push_back((static_cast<RooAbsRealLValue const&>(_x.arg())).getMin(_rangeName.Data())+1e-6) ;
  hint->push_back((static_cast<RooAbsRealLValue const&>(_x.arg())).getMax(_rangeName.Data())-1e-6) ;
  hint->push_back((static_cast<RooAbsRealLValue const&>(_x.arg())).getMax(_rangeName.Data())+1e-6) ;
  return hint ;
}

