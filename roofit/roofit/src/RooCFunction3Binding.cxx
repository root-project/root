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

/** \class RooCFunction3Binding
    \ingroup Roofit

RooCFunction3Binding is a templated implementation of class RooAbsReal that binds
generic C(++) functions to a RooAbsReal argument thus allowing generic C++
functions to be used as RooFit functions. Instances of function binding
classes are fully functional RooFit function objects with one exception:
if the bound function is _not_ a standard TMath or MathMore function the
class cannot be persisted in a RooWorkspace without registering the function
pointer first using RooCFunction3Binding<T1,T2,T3,T4>::register().
**/

#include "Riostream.h"
#include "RooCFunction3Binding.h"

using namespace std ;

#ifndef ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
#define ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
templateClassImp(RooCFunction3Binding);
templateClassImp(RooCFunction3Ref);
#endif


namespace RooFit {

  RooAbsReal* bindFunction(const char* name,CFUNCD3DDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3Binding<Double_t,Double_t,Double_t,Double_t>(name,name,func,x,y,z) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD3DDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3Binding<Double_t,Double_t,Double_t,bool>(name,name,func,x,y,z) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD3DII func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3Binding<Double_t,Double_t,Int_t,Int_t>(name,name,func,x,y,z) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD3UDU func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3Binding<Double_t,UInt_t,Double_t,UInt_t>(name,name,func,x,y,z) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD3UDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3Binding<Double_t,UInt_t,Double_t,Double_t>(name,name,func,x,y,z) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD3UUD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3Binding<Double_t,UInt_t,UInt_t,Double_t>(name,name,func,x,y,z) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD3DDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3PdfBinding<Double_t,Double_t,Double_t,Double_t>(name,name,func,x,y,z) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD3DDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3PdfBinding<Double_t,Double_t,Double_t,bool>(name,name,func,x,y,z) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD3DII func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3PdfBinding<Double_t,Double_t,Int_t,Int_t>(name,name,func,x,y,z) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD3UDU func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3PdfBinding<Double_t,UInt_t,Double_t,UInt_t>(name,name,func,x,y,z) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD3UDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3PdfBinding<Double_t,UInt_t,Double_t,Double_t>(name,name,func,x,y,z) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD3UUD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooCFunction3PdfBinding<Double_t,UInt_t,UInt_t,Double_t>(name,name,func,x,y,z) ;
  }

}
