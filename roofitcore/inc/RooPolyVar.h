/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooPolyVar.rdl,v 1.3 2002/09/05 04:33:48 verkerke Exp $
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
#ifndef ROO_POLY_VAR
#define ROO_POLY_VAR

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooListProxy.hh"

class RooRealVar;
class RooArgList ;

class RooPolyVar : public RooAbsReal {
public:

  RooPolyVar() ;
  RooPolyVar(const char* name, const char* title, RooAbsReal& x) ;
  RooPolyVar(const char *name, const char *title,
		RooAbsReal& _x, const RooArgList& _coefList, Int_t lowestOrder=0) ;

  RooPolyVar(const RooPolyVar& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooPolyVar(*this, newname); }
  inline virtual ~RooPolyVar() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;

protected:

  RooRealProxy _x;
  RooListProxy _coefList ;
  Int_t _lowestOrder ;
  TIterator* _coefIter ;  //! do not persist

  Double_t evaluate() const;

  ClassDef(RooPolyVar,1) // Polynomial PDF
};

#endif
