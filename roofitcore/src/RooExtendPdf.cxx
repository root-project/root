/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooExtendPdf.cxx,v 1.21 2007/05/11 10:14:56 verkerke Exp $
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

#include "RooFit.h"

#include "RooExtendPdf.h"
#include "RooExtendPdf.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"
#include "RooNameReg.h"

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

  if (_rangeName && (!nset || nset->getSize()==0)) {
    cout << "RooExtendPdf::expectedEvents(" << GetName() << ") WARNING: RooExtendPdf needs non-null normalization set to calculate fraction in range " 
	 << _rangeName << ".  Results may be nonsensical" << endl ;  
  }

  Double_t nExp = _n ;

  // Optionally multiply with fractional normalization
  if (_rangeName) {

    globalSelectComp(kTRUE) ;
    Double_t fracInt = pdf.getNormObj(nset,nset,_rangeName)->getVal() ;
    globalSelectComp(kFALSE) ;


    if ( fracInt == 0. || _n == 0.) {
      cout << "RooExtendPdf(" << GetName() << ") WARNING: nExpected = " << _n << " / " 
	   << fracInt << " for nset = " ;
      if (nset) nset->Print("1") ; else cout << "<none>" << endl ;
    }

    nExp /= fracInt ;    


    // cout << "RooExtendPdf::expectedEvents(" << GetName() << ") fracInt = " << fracInt << " _n = " << _n << " nExpect = " << nExp << endl ;

  }

  // Multiply with original Nexpected, if defined
  if (pdf.canBeExtended()) nExp *= pdf.expectedEvents(nset) ;

  return nExp ;
}



