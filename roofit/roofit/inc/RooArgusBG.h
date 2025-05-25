/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooArgusBG.h,v 1.13 2007/07/12 20:30:49 wouter Exp $
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
#ifndef ROO_ARGUS_BG
#define ROO_ARGUS_BG

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooArgusBG : public RooAbsPdf {
public:
  RooArgusBG() {} ;
  // One of the original constructors without RooAbsReal::Ref for backwards compatibility.
  inline RooArgusBG(const char *name, const char *title,
        RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c, RooAbsReal& _p)
      : RooArgusBG{name, title, RooAbsReal::Ref{_m}, RooAbsReal::Ref{_m0}, RooAbsReal::Ref{_c}, RooAbsReal::Ref{_p}} {}
  // One of the original constructors without RooAbsReal::Ref for backwards compatibility.
  inline RooArgusBG(const char *name, const char *title,
        RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c)
      : RooArgusBG{name, title, RooAbsReal::Ref{_m}, RooAbsReal::Ref{_m0}, RooAbsReal::Ref{_c}} {}
  RooArgusBG(const char *name, const char *title,
        RooAbsReal::Ref _m, RooAbsReal::Ref _m0, RooAbsReal::Ref _c, RooAbsReal::Ref _p=0.5);
  RooArgusBG(const RooArgusBG& other,const char* name=nullptr) ;
  TObject* clone(const char* newname=nullptr) const override { return new RooArgusBG(*this,newname); }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;

protected:
  RooRealProxy m ;
  RooRealProxy m0 ;
  RooRealProxy c ;
  RooRealProxy p ;

  double evaluate() const override ;
  void doEval(RooFit::EvalContext &) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }


//   void initGenerator();

private:
  ClassDefOverride(RooArgusBG,1) // Argus background shape PDF
};

#endif
