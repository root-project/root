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

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooConvCoefVar is an auxilary class that represents the coefficient
// of a RooAbsAnaConvPdf implementation as a separate RooAbsReal object
// to be able to interface these coefficient terms with the generic
// RooRealIntegral integration mechanism
// END_HTML
//
//

#include "RooFit.h"

#include "RooAbsAnaConvPdf.h"
#include "RooAbsAnaConvPdf.h"
#include "RooConvCoefVar.h"

using namespace std;

ClassImp(RooConvCoefVar)
;


//_____________________________________________________________________________
RooConvCoefVar::RooConvCoefVar(const char *name, const char *title, const RooAbsAnaConvPdf& input, 
			       Int_t coefIdx, const RooArgSet* varList) :
  RooAbsReal(name,title),
  _varSet("varSet","Set of coefficient variables",this),
  _convPdf("convPdf","Convoluted PDF",this,(RooAbsReal&)input,kFALSE,kFALSE),
  _coefIdx(coefIdx)
{
  // Constuctor given a RooAbsAnaConvPdf a coefficient index and a set with the
  // convoluted observable(s)
  if (varList) _varSet.add(*varList) ;
}



//_____________________________________________________________________________
RooConvCoefVar::RooConvCoefVar(const RooConvCoefVar& other, const char* name) :
  RooAbsReal(other,name),
  _varSet("varSet",this,other._varSet),
  _convPdf("convPdf",this,other._convPdf),
  _coefIdx(other._coefIdx)
{
  // Copy constructor
}



//_____________________________________________________________________________
Double_t RooConvCoefVar::getValV(const RooArgSet*) const 
{ 
  // Return value of chosen coefficient
  return evaluate() ; 
}



//_____________________________________________________________________________
Double_t RooConvCoefVar::evaluate() const 
{
  // Return value of chosen coefficient
  return ((RooAbsAnaConvPdf&)_convPdf.arg()).coefficient(_coefIdx) ;
}



//_____________________________________________________________________________
Int_t RooConvCoefVar::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName) const 
{
  // Return analytical integration capabilities of chosen coefficient

  Int_t code = ((RooAbsAnaConvPdf&)_convPdf.arg()).getCoefAnalyticalIntegral(_coefIdx,allVars,analVars,rangeName) ;
  return code ;
}



//_____________________________________________________________________________
Double_t RooConvCoefVar::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  // Return analytical integral of chosen coefficient

  return ((RooAbsAnaConvPdf&)_convPdf.arg()).coefAnalyticalIntegral(_coefIdx,code,rangeName) ;
}

