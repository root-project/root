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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooCFunction4Binding is a templated implementation of class RooAbsReal that binds 
// generic C(++) functions to a RooAbsReal argument thus allowing generic C++
// functions to be used as RooFit functions. Instances of function binding
// classes are fully functional RooFit function objects with one exception:
// if the bound function is _not_ a standard TMath or MathMore function the
// class cannot be persisted in a RooWorkspace without registering the function
// pointer first using RooCFunction4Binding<T1,T2,T3,T4>::register().
// END_HTML
//

#include "Riostream.h" 
#include "RooCFunction4Binding.h" 
#include "RooCintUtils.h"

using namespace std ;

#ifndef ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
#define ROOFIT_R__NO_CLASS_TEMPLATE_SPECIALIZATION
templateClassImp(RooCFunction4Binding) 
templateClassImp(RooCFunction4Ref) 
#endif 


template<> RooCFunction4Map<double,double,double,double,double>* RooCFunction4Ref<double,double,double,double,double>::_fmap = 0 ;
template<> RooCFunction4Map<double,double,double,double,int>* RooCFunction4Ref<double,double,double,double,int>::_fmap = 0 ;
template<> RooCFunction4Map<double,double,double,double,bool>* RooCFunction4Ref<double,double,double,double,bool>::_fmap = 0 ;


namespace RooFit {

  RooAbsReal* bindFunction(const char* name,CFUNCD4DDDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) {
    return new RooCFunction4Binding<Double_t,Double_t,Double_t,Double_t,Double_t>(name,name,func,x,y,z,w) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD4DDDI func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) {
    return new RooCFunction4Binding<Double_t,Double_t,Double_t,Double_t,Int_t>(name,name,func,x,y,z,w) ;
  }

  RooAbsReal* bindFunction(const char* name,CFUNCD4DDDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) {
    return new RooCFunction4Binding<Double_t,Double_t,Double_t,Double_t,Bool_t>(name,name,func,x,y,z,w) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD4DDDD func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) {
    return new RooCFunction4PdfBinding<Double_t,Double_t,Double_t,Double_t,Double_t>(name,name,func,x,y,z,w) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD4DDDI func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) {
    return new RooCFunction4PdfBinding<Double_t,Double_t,Double_t,Double_t,Int_t>(name,name,func,x,y,z,w) ;
  }

  RooAbsPdf* bindPdf(const char* name,CFUNCD4DDDB func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z, RooAbsReal& w) {
    return new RooCFunction4PdfBinding<Double_t,Double_t,Double_t,Double_t,Bool_t>(name,name,func,x,y,z,w) ;
  }

}
