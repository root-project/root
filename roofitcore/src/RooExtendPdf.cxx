/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooExtendPdf.cc,v 1.13 2004/11/29 20:23:35 wverkerke Exp $
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

// -- CLASS DESCRIPTION [PDF] --
//  RooExtendPdf is a wrappper around an existing PDF that adds a 
//  parameteric extended likelihood term to the PDF, optionally multiplied by a 
//  fractional term from a partial normalization of the PDF:
//
//  nExpected = N   _or Expected = N * frac 
//
//  where N is supplied as a RooAbsReal to RooExtendPdf.
//  The fractional term is defined as
//                          _       _ _   _  _
//            Int(cutRegion[x]) pdf(x,y) dx dy 
//     frac = ---------------_-------_-_---_--_ 
//            Int(normRegion[x]) pdf(x,y) dx dy 
//
//        _                                                               _
//  where x is the set of dependents involved in the selection region and y
//  is the set of remaining dependents.
//            _
//  cutRegion[x] is an limited integration range that is contained in
//  the nominal integration range normRegion[x[]
//

#include "RooFitCore/RooExtendPdf.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooNameReg.hh"
using std::cout;
using std::endl;

ClassImp(RooExtendPdf)
;


RooExtendPdf::RooExtendPdf(const char *name, const char *title, const RooAbsPdf& pdf,
			   const RooAbsReal& norm, const char* rangeName) :
  RooAbsPdf(name,title),
  _pdf("pdf","PDF",this,(RooAbsReal&)pdf),
  _n("n","Normalization",this,(RooAbsReal&)norm),
  _rangeName(RooNameReg::ptr(rangeName))
{
  // Constructor. The ExtendedPdf behaves identical to the supplied input pdf,
  // but adds an extended likelihood term. The expected number of events return
  // is 'norm'. If a rangename is given, the number of events is interpreted as
  // the number of events in the given range

  // Copy various setting from pdf
  setUnit(_pdf.arg().getUnit()) ;
  setPlotLabel(_pdf.arg().getPlotLabel()) ;
}



RooExtendPdf::RooExtendPdf(const RooExtendPdf& other, const char* name) :
  RooAbsPdf(other,name),
  _pdf("pdf",this,other._pdf),
  _n("n",this,other._n),
  _rangeName(other._rangeName)
{
  // Copy constructor
}


RooExtendPdf::~RooExtendPdf() 
{
  // Destructor

}



Double_t RooExtendPdf::expectedEvents(const RooArgSet* nset) const 
{
  // Return the number of expected events, which is
  //
  // n / [ Int(xC,yF) pdf(x,y) / Int(xF,yF) pdf(x,y) ]
  //
  // Where x is the set of dependents with cuts defined
  // and y are the other dependents. xC is the integration
  // of x over the cut range, xF is the integration of
  // x over the full range.

  RooAbsPdf& pdf = (RooAbsPdf&)_pdf.arg() ;

  Double_t nExp = _n ;

  // Optionally multiply with fractional normalization
  if (_rangeName) {

    Double_t normInt = pdf.getNorm(nset) ;
    
    // Evaluate fraction integral and return normalized by full integral
    Double_t fracInt = pdf.getNormObj(nset,_rangeName)->getVal() ;

    if ( fracInt == 0. || normInt == 0. || _n == 0.) {
      cout << "RooExtendPdf(" << GetName() << ") WARNING: nExpected = " << _n << " / ( " 
	   << fracInt << " / " << normInt << " ), for nset = " ;
      if (nset) nset->Print("1") ; else cout << "<none>" << endl ;
    }

    nExp /= (fracInt / normInt) ;
  }

  // Multiply with original Nexpected, if defined
  if (pdf.canBeExtended()) nExp *= pdf.expectedEvents(nset) ;

  return nExp ;
}



