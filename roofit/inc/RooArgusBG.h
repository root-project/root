/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooArgusBG.rdl,v 1.8 2003/05/14 05:30:31 wverkerke Exp $
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
#ifndef ROO_ARGUS_BG
#define ROO_ARGUS_BG

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;
class RooAbsReal;

class RooArgusBG : public RooAbsPdf {
public:
  RooArgusBG(const char *name, const char *title, 
	     RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c);
  RooArgusBG(const char *name, const char *title, 
	     RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c, RooAbsReal& _p);
  RooArgusBG(const RooArgusBG& other,const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooArgusBG(*this,newname); }
  inline virtual ~RooArgusBG() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  Double_t analyticalIntegral(Int_t code) const ;

protected:
  RooRealProxy m ;
  RooRealProxy m0 ;
  RooRealProxy c ;
  RooRealProxy p ;

  Double_t evaluate() const ;
//   void initGenerator();

private:
  ClassDef(RooArgusBG,0) // Argus background shape PDF
};

#endif
