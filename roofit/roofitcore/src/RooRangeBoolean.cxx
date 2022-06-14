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

RooRangeBoolean
**/

#include "Riostream.h"
#include <math.h>

#include "RooRangeBoolean.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooMsgService.h"

using namespace std;

ClassImp(RooRangeBoolean);


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooRangeBoolean::RooRangeBoolean()
{
}


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
/// Destructor

RooRangeBoolean::~RooRangeBoolean()
{
}




////////////////////////////////////////////////////////////////////////////////
/// Return 1 if x is in range, zero otherwis

double RooRangeBoolean::evaluate() const
{
  double xmin = ((RooAbsRealLValue&)_x.arg()).getMin(_rangeName.Data()) ;
  double xmax = ((RooAbsRealLValue&)_x.arg()).getMax(_rangeName.Data()) ;

  double ret = (_x >= xmin && _x < xmax) ? 1.0 : 0.0 ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////

std::list<double>* RooRangeBoolean::plotSamplingHint(RooAbsRealLValue& obs, double /*xlo*/, double /*xhi*/) const
{
  if (string(obs.GetName())!=_x.arg().GetName()) {
    return 0 ;
  }

  list<double>* hint = new list<double> ;
  hint->push_back(((RooAbsRealLValue&)_x.arg()).getMin(_rangeName.Data())-1e-6) ;
  hint->push_back(((RooAbsRealLValue&)_x.arg()).getMin(_rangeName.Data())+1e-6) ;
  hint->push_back(((RooAbsRealLValue&)_x.arg()).getMax(_rangeName.Data())-1e-6) ;
  hint->push_back(((RooAbsRealLValue&)_x.arg()).getMax(_rangeName.Data())+1e-6) ;
  return hint ;
}

