/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id$
 * Authors:
 *   GR, Gerhard Raven, UC, San Diego, Gerhard.Raven@slac.stanford.edu
 * History:
 *   20-Oct-2001 GR Created initial version from RooPolynomial
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooChebychev.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooChebychev)
;

RooChebychev::RooChebychev()
{
}

RooChebychev::RooChebychev(const char* name, const char* title, 
                           RooAbsReal& x, const RooArgList& coefList): 
  RooAbsPdf(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this)
{
  // Constructor
  TIterator* coefIter = coefList.createIterator() ;
  RooAbsArg* coef ;
  while(coef = (RooAbsArg*)coefIter->Next()) {
    if (!dynamic_cast<RooAbsReal*>(coef)) {
      cout << "RooChebychev::ctor(" << GetName() << ") ERROR: coefficient " << coef->GetName() 
	   << " is not of type RooAbsReal" << endl ;
      assert(0) ;
    }
    _coefList.add(*coef) ;
  }
  delete coefIter ;
}


RooChebychev::RooChebychev(const RooChebychev& other, const char* name) :
  RooAbsPdf(other, name), 
  _x("x", this, other._x), 
  _coefList("coefList",this,other._coefList)
{
}

inline static double p0(double t,double a) {  return a; }
inline static double p1(double t,double a,double b) {  return a*t+b; }
inline static double p2(double t,double a,double b,double c) {  return p1(t,p1(t,a,b),c); }
inline static double p3(double t,double a,double b,double c,double d) {  return p2(t,p1(t,a,b),c,d); }
inline static double p4(double t,double a,double b,double c,double d,double e) {  return p3(t,p1(t,a,b),c,d,e); }

Double_t RooChebychev::evaluate() const 
{

  Double_t xmin = _x.min(); Double_t xmax = _x.max();
  Double_t x(-1+2*(_x-xmin)/(xmax-xmin));
  Double_t x2(x*x);
  Double_t sum(0) ;
  switch (_coefList.getSize()) {
             case  7: sum+=((RooAbsReal&)_coefList[6]).getVal()*x*p3(x2,64,-112,56,-7);
             case  6: sum+=((RooAbsReal&)_coefList[5]).getVal()*p3(x2,32,-48,18,-1);
             case  5: sum+=((RooAbsReal&)_coefList[4]).getVal()*x*p2(x2,16,-20,5);
             case  4: sum+=((RooAbsReal&)_coefList[3]).getVal()*p2(x2,8,-8,1);
             case  3: sum+=((RooAbsReal&)_coefList[2]).getVal()*x*p1(x2,4,-3);
             case  2: sum+=((RooAbsReal&)_coefList[1]).getVal()*p1(x2,2,-1);
             case  1: sum+=((RooAbsReal&)_coefList[0]).getVal()*x;
             case  0: sum+=1;
  }
  return sum;
}

Int_t RooChebychev::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars, analVars, _x)) return 1;
  return 0;
}

Double_t RooChebychev::analyticalIntegral(Int_t code) const 
{
  assert(code==1) ;
  Double_t xmin = _x.min(); Double_t xmax = _x.max();
  Double_t norm(0) ;
  switch(_coefList.getSize()) {
    case  7: case  6: norm+=((RooAbsReal&)_coefList[5]).getVal()*(-1 + 18./3. - 48./5. + 32./7.);
    case  5: case  4: norm+=((RooAbsReal&)_coefList[3]).getVal()*( 1 -  8./3. +  8./5.);
    case  3: case  2: norm+=((RooAbsReal&)_coefList[1]).getVal()*(-1 +  2./3.);
    case  1: case  0: norm+= 1;
  }
  norm *= xmax-xmin;
  return norm;
}
