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
\file RooMoment.cxx
\class RooMoment
\ingroup Roofitcore
**/


#include "Riostream.h"
#include <cmath>

#include "RooMoment.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooFunctor.h"
#include "RooFormulaVar.h"
#include "RooGlobalFunc.h"
#include "RooConstVar.h"
#include "RooRealIntegral.h"
#include "RooNumIntConfig.h"
#include <string>


////////////////////////////////////////////////////////////////////////////////

RooMoment::RooMoment(const char* name, const char* title, RooAbsReal& func, RooRealVar& x, Int_t orderIn, bool centr, bool takeRoot) :
  RooAbsMoment(name, title,func,x,orderIn,takeRoot),
  _xf("!xf","xf",this,false,false),
  _ixf("!ixf","ixf",this),
  _if("!if","if",this)
{
  setExpensiveObjectCache(func.expensiveObjectCache()) ;

  std::string pname=Form("%s_product",name) ;

  std::unique_ptr<RooFormulaVar> XF;
  if (centr) {
    std::string formula=Form("pow((@0-@1),%d)*@2",_order) ;
    std::string m1name=Form("%s_moment1",GetName()) ;
    RooAbsReal* mom1 = func.mean(x) ;
    XF = std::make_unique<RooFormulaVar>(pname.c_str(),formula.c_str(),RooArgList(x,*mom1,func)) ;
    XF->setExpensiveObjectCache(func.expensiveObjectCache()) ;
    addOwnedComponents(*mom1) ;
    _mean.setArg(*mom1) ;
  } else {
    std::string formula=Form("pow(@0,%d)*@1",_order) ;
    XF = std::make_unique<RooFormulaVar>(pname.c_str(),formula.c_str(),RooArgSet(x,func)) ;
    XF->setExpensiveObjectCache(func.expensiveObjectCache()) ;
  }

  if (func.isBinnedDistribution(x)) {
    XF->specialIntegratorConfig(true)->method1D().setLabel("RooBinIntegrator");
  }

  std::unique_ptr<RooAbsReal> intXF{XF->createIntegral(x)};
  std::unique_ptr<RooAbsReal> intF{func.createIntegral(x)};
  static_cast<RooRealIntegral&>(*intXF).setCacheNumeric(true) ;
  static_cast<RooRealIntegral&>(*intF).setCacheNumeric(true) ;

  _xf.setArg(*XF) ;
  _ixf.setArg(*intXF) ;
  _if.setArg(*intF) ;
  addOwnedComponents(std::move(XF)) ;
  addOwnedComponents(std::move(intXF));
  addOwnedComponents(std::move(intF));
}

////////////////////////////////////////////////////////////////////////////////

RooMoment::RooMoment(const char* name, const char* title, RooAbsReal& func, RooRealVar& x, const RooArgSet& nset,
           Int_t orderIn, bool centr, bool takeRoot, bool intNSet) :
  RooAbsMoment(name, title,func,x,orderIn,takeRoot),
  _xf("!xf","xf",this,false,false),
  _ixf("!ixf","ixf",this),
  _if("!if","if",this)
{
  setExpensiveObjectCache(func.expensiveObjectCache()) ;

  _nset.add(nset) ;

  std::string pname=Form("%s_product",name) ;
  std::unique_ptr<RooFormulaVar> XF;
  if (centr) {
    std::string formula=Form("pow((@0-@1),%d)*@2",_order) ;
    std::string m1name=Form("%s_moment1",GetName()) ;
    RooAbsReal* mom1 = func.mean(x,nset) ;
    XF = std::make_unique<RooFormulaVar>(pname.c_str(),formula.c_str(),RooArgList(x,*mom1,func)) ;
    XF->setExpensiveObjectCache(func.expensiveObjectCache()) ;
    addOwnedComponents(*mom1) ;
    _mean.setArg(*mom1) ;
  } else {
    std::string formula=Form("pow(@0,%d)*@1",_order) ;
    XF = std::make_unique<RooFormulaVar>(pname.c_str(),formula.c_str(),RooArgSet(x,func)) ;
    XF->setExpensiveObjectCache(func.expensiveObjectCache()) ;
  }

  if (func.isBinnedDistribution(x)) {
    XF->specialIntegratorConfig(true)->method1D().setLabel("RooBinIntegrator");
  }

  RooArgSet intSet(x) ;
  if (intNSet) intSet.add(_nset,true) ;

  std::unique_ptr<RooAbsReal> intXF{XF->createIntegral(intSet, &_nset)};
  std::unique_ptr<RooAbsReal> intF{func.createIntegral(intSet, &_nset)};
  static_cast<RooRealIntegral&>(*intXF).setCacheNumeric(true) ;
  static_cast<RooRealIntegral&>(*intF).setCacheNumeric(true) ;

  _xf.setArg(*XF) ;
  _ixf.setArg(*intXF) ;
  _if.setArg(*intF) ;
  addOwnedComponents(std::move(XF)) ;
  addOwnedComponents(std::move(intXF));
  addOwnedComponents(std::move(intF));
}



////////////////////////////////////////////////////////////////////////////////

RooMoment::RooMoment(const RooMoment& other, const char* name) :
  RooAbsMoment(other, name),
  _xf("xf",this,other._xf),
  _ixf("ixf",this,other._ixf),
  _if("if",this,other._if)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate value

double RooMoment::evaluate() const
{
  double ratio = _ixf / _if ;
  double ret =  _takeRoot ? pow(ratio,1.0/_order) : ratio ;
  return ret ;
}
