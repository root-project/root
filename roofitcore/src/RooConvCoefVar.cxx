/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConvCoefVar.cc,v 1.4 2003/05/14 02:58:40 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooConvCoefVar is an auxilary class that represents the coefficient
// of a RooConvolutedPdf implementation as a separate RooAbsReal object
// to be able to interface these coefficient terms with RooRealIntegreal
//

#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooConvCoefVar.hh"

ClassImp(RooConvCoefVar)
;

RooConvCoefVar::RooConvCoefVar(const char *name, const char *title, const RooConvolutedPdf& input, 
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


Double_t RooConvCoefVar::evaluate() const 
{
  return ((RooConvolutedPdf&)_convPdf.arg()).coefficient(_coefIdx) ;
}

Int_t RooConvCoefVar::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  Int_t code = ((RooConvolutedPdf&)_convPdf.arg()).getCoefAnalyticalIntegral(allVars,analVars) ;
//   cout << "RooConvCoefVar::getAnalyticalIntegral code = " << code << " for " ; analVars.Print("1") ;
  return code ;
}


Double_t RooConvCoefVar::analyticalIntegral(Int_t code) const 
{
//   cout << "RooConvCoefVar::analyticalIntegral(" << _coefIdx << "," << code << ")" << endl ;
  return ((RooConvolutedPdf&)_convPdf.arg()).coefAnalyticalIntegral(_coefIdx,code) ;
}

