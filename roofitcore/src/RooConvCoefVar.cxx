/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooConvCoefVar.cc,v 1.2 2001/11/19 07:23:55 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   09-Oct-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
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
  _convPdf("convPdf","Convoluted PDF",this,(RooAbsReal&)input),
  _varSet("varSet","Set of coefficient variables",this),
  _coefIdx(coefIdx)
{
  // Constuctor
  if (varList) _varSet.add(*varList) ;
}


RooConvCoefVar::RooConvCoefVar(const RooConvCoefVar& other, const char* name) :
  RooAbsReal(other,name),
  _convPdf("convPdf",this,other._convPdf),
  _varSet("varSet",this,other._varSet),
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

