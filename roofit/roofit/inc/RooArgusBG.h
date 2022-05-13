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
  RooArgusBG(const char *name, const char *title,
        RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c);
  RooArgusBG(const char *name, const char *title,
        RooAbsReal& _m, RooAbsReal& _m0, RooAbsReal& _c, RooAbsReal& _p);
  RooArgusBG(const RooArgusBG& other,const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooArgusBG(*this,newname); }
  inline ~RooArgusBG() override { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

protected:
  RooRealProxy m ;
  RooRealProxy m0 ;
  RooRealProxy c ;
  RooRealProxy p ;

  double evaluate() const override ;
  void computeBatch(cudaStream_t*, double* output, size_t size, RooBatchCompute::DataMap&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }


//   void initGenerator();

private:
  ClassDefOverride(RooArgusBG,1) // Argus background shape PDF
};

#endif
