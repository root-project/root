/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooConvCoefVar.cxx,v 1.13 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [REAL] --
// RooConvCoefVar is an auxilary class that represents the coefficient
// of a RooAbsAnaConvPdf implementation as a separate RooAbsReal object
// to be able to interface these coefficient terms with RooRealIntegreal
//

#include "RooFit.h"

#include "RooAbsAnaConvPdf.h"
#include "RooAbsAnaConvPdf.h"
#include "RooConvCoefVar.h"

ClassImp(RooConvCoefVar)
;

RooConvCoefVar::RooConvCoefVar(const char *name, const char *title, const RooAbsAnaConvPdf& input, 
			       Int_t coefIdx, const RooArgSet* varList) :
  RooAbsReal(name,title),
  _varSet("varSet","Set of coefficient variables",this),
  _convPdf("convPdf","Convoluted PDF",this,(RooAbsReal&)input),
  _coefIdx(coefIdx)
{
  // Constuctor
  if (varList) _varSet.add(*varList) ;
}


RooConvCoefVar::RooConvCoefVar(const RooConvCoefVar& other, const char* name) :
  RooAbsReal(other,name),
  _varSet("varSet",this,other._varSet),
  _convPdf("convPdf",this,other._convPdf),
  _coefIdx(other._coefIdx)
{
  // Copy constructor
}


Double_t RooConvCoefVar::getVal(const RooArgSet*) const 
{ 
  return evaluate() ; 
}


Double_t RooConvCoefVar::evaluate() const 
{
  return ((RooAbsAnaConvPdf&)_convPdf.arg()).coefficient(_coefIdx) ;
}

Int_t RooConvCoefVar::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const 
{
  Int_t code = ((RooAbsAnaConvPdf&)_convPdf.arg()).getCoefAnalyticalIntegral(allVars,analVars,rangeName) ;
//   cout << "RooConvCoefVar::getAnalyticalIntegral code = " << code << " for " ; analVars.Print("1") ;
  return code ;
}


Double_t RooConvCoefVar::analyticalIntegral(Int_t code, const char* rangeName) const 
{
//   cout << "RooConvCoefVar::analyticalIntegral(" << _coefIdx << "," << code << ")" << endl ;
  return ((RooAbsAnaConvPdf&)_convPdf.arg()).coefAnalyticalIntegral(_coefIdx,code,rangeName) ;
}

