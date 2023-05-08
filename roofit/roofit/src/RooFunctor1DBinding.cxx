/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, NIKHEF, verkerke@nikhef.nl                         *
 *                                                                           *
 * Copyright (c) 2000-2008, NIKHEF, Regents of the University of California  *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 *****************************************************************************/

/** \class RooFunctor1DBinding
    \ingroup Roofit

RooCFunction1Binding is a templated implementation of class RooAbsReal that binds
generic C(++) functions to a RooAbsReal argument thus allowing generic C++
functions to be used as RooFit functions. Instances of function binding
classes are fully functional RooFit function objects with one exception:
if the bound function is _not_ a standard TMath or MathMore function the
class cannot be persisted in a RooWorkspace without registering the function
pointer first using RooCFunction1Binding<T1,T2>::register().
**/

/** \class RooFunctor1DPdfBinding
    \ingroup Roofit
**/

#include "Riostream.h"
#include "RooFunctor1DBinding.h"

using namespace std ;

ClassImp(RooFunctor1DBinding);
ClassImp(RooFunctor1DPdfBinding);

////////////////////////////////////////////////////////////////////////////////

RooFunctor1DBinding::RooFunctor1DBinding(const char *name, const char *title, const ROOT::Math::IBaseFunctionOneDim& ftor, RooAbsReal& x) :
  RooAbsReal(name,title),
  func(&ftor),
  var("x","x",this,x)
{
}

////////////////////////////////////////////////////////////////////////////////

RooFunctor1DBinding::RooFunctor1DBinding(const RooFunctor1DBinding& other, const char* name) :
  RooAbsReal(other,name),
  func(other.func),
  var("x",this,other.var)
{
  // Copy constructor
}

////////////////////////////////////////////////////////////////////////////////

void RooFunctor1DBinding::printArgs(ostream& os) const {
  // Print object arguments and name/address of function pointer
  os << "[ function=" << func << " " ;
  for (Int_t i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* p = getProxy(i) ;
    if (!TString(p->name()).BeginsWith("!")) {
      p->print(os) ;
      os << " " ;
    }
  }
  os << "]" ;
}

////////////////////////////////////////////////////////////////////////////////

double RooFunctor1DBinding::evaluate() const {
    // Return value of embedded function using value of referenced variable x
    return (*func)(var.arg().getVal()) ;
  }

////////////////////////////////////////////////////////////////////////////////

RooFunctor1DPdfBinding::RooFunctor1DPdfBinding(const char *name, const char *title, const ROOT::Math::IBaseFunctionOneDim& ftor, RooAbsReal& x) :
  RooAbsPdf(name,title),
  func(&ftor),
  var("x","x",this,x)
{
}

////////////////////////////////////////////////////////////////////////////////

RooFunctor1DPdfBinding::RooFunctor1DPdfBinding(const RooFunctor1DPdfBinding& other, const char* name) :
  RooAbsPdf(other,name),
  func(other.func),
  var("x",this,other.var)
{
  // Copy constructor
}

////////////////////////////////////////////////////////////////////////////////

void RooFunctor1DPdfBinding::printArgs(ostream& os) const {
  // Print object arguments and name/address of function pointer
  os << "[ function=" << func << " " ;
  for (Int_t i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* p = getProxy(i) ;
    if (!TString(p->name()).BeginsWith("!")) {
      p->print(os) ;
      os << " " ;
    }
  }
  os << "]" ;
}

////////////////////////////////////////////////////////////////////////////////

double RooFunctor1DPdfBinding::evaluate() const {
    // Return value of embedded function using value of referenced variable x
    return (*func)(var.arg().getVal()) ;
  }

namespace RooFit {

  RooAbsReal* bindFunction(const char* name, const ROOT::Math::IBaseFunctionOneDim& ftor, RooAbsReal& var) {
    return new RooFunctor1DBinding(name,name,ftor,var) ;
  }

  RooAbsPdf*  bindPdf(const char* name, const ROOT::Math::IBaseFunctionOneDim& ftor, RooAbsReal& var) {
    return new RooFunctor1DPdfBinding(name,name,ftor,var) ;
  }

}
