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
\file RooFirstMoment.cxx
\class RooFirstMoment
\ingroup Roofitcore
**/

#include <RooFirstMoment.h>
#include <RooAbsReal.h>
#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooMsgService.h>
#include <RooRealVar.h>
#include <RooGlobalFunc.h>
#include <RooRealIntegral.h>
#include <RooNumIntConfig.h>
#include <RooProduct.h>

#include <Riostream.h>

#include <cmath>
#include <string>

ClassImp(RooFirstMoment);

////////////////////////////////////////////////////////////////////////////////

RooFirstMoment::RooFirstMoment(const char* name, const char* title, RooAbsReal& func, RooRealVar& x) :
  RooAbsMoment(name, title,func,x,1,false),
  _xf("!xf","xf",this,false,false),
  _ixf("!ixf","ixf",this),
  _if("!if","if",this)
{
  setExpensiveObjectCache(func.expensiveObjectCache()) ;

  std::string pname = std::string(name) + "_product";

  auto XF = std::make_unique<RooProduct>(pname.c_str(),pname.c_str(),RooArgSet(x,func));
  XF->setExpensiveObjectCache(func.expensiveObjectCache()) ;

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

RooFirstMoment::RooFirstMoment(const char* name, const char* title, RooAbsReal& func, RooRealVar& x, const RooArgSet& nset, bool intNSet) :
  RooAbsMoment(name, title,func,x,1,false),
  _xf("!xf","xf",this,false,false),
  _ixf("!ixf","ixf",this),
  _if("!if","if",this)
{
  setExpensiveObjectCache(func.expensiveObjectCache()) ;

  _nset.add(nset) ;

  std::string pname = std::string(name) + "_product";

  auto XF = std::make_unique<RooProduct>(pname.c_str(),pname.c_str(),RooArgSet(x,func)) ;
  XF->setExpensiveObjectCache(func.expensiveObjectCache()) ;

  if (func.isBinnedDistribution(x)) {
    XF->specialIntegratorConfig(true)->method1D().setLabel("RooBinIntegrator");
  }

  if (intNSet && !_nset.empty() && func.isBinnedDistribution(_nset)) {
    XF->specialIntegratorConfig(true)->method2D().setLabel("RooBinIntegrator");
    XF->specialIntegratorConfig(true)->methodND().setLabel("RooBinIntegrator");
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

RooFirstMoment::RooFirstMoment(const RooFirstMoment& other, const char* name) :
  RooAbsMoment(other, name),
  _xf("xf",this,other._xf),
  _ixf("ixf",this,other._ixf),
  _if("if",this,other._if)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate value

double RooFirstMoment::evaluate() const
{
  double ratio = _ixf / _if ;
  //cout << "\nRooFirstMoment::eval(" << GetName() << ") val = " << ratio << std::endl ;
  return ratio ;
}
