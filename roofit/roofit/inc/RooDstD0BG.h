/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooDstD0BG.h,v 1.12 2007/07/12 20:30:49 wouter Exp $
 * Authors:                                                                  *
 *   UE, Ulrik Egede,     RAL,               U.Egede@rl.ac.uk                *
 *   MT, Max Turri,       UC Santa Cruz      turri@slac.stanford.edu         *
 *   CC, Chih-hsiang Cheng, Stanford         chcheng@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          RAL and Stanford University. All rights reserved.*
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_DstD0_BG
#define ROO_DstD0_BG

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;

class RooDstD0BG : public RooAbsPdf {
public:
  RooDstD0BG() {} ;
  RooDstD0BG(const char *name, const char *title,
        RooAbsReal& _dm, RooAbsReal& _dm0, RooAbsReal& _c,
        RooAbsReal& _a, RooAbsReal& _b);

  RooDstD0BG(const RooDstD0BG& other, const char *name=0) ;
  TObject *clone(const char *newname) const override {
    return new RooDstD0BG(*this,newname); }
  inline ~RooDstD0BG() override { };

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

protected:

  RooRealProxy dm ;
  RooRealProxy dm0 ;
  RooRealProxy C,A,B ;

  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooBatchCompute::DataMap&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

private:

  ClassDefOverride(RooDstD0BG,1) // D*-D0 mass difference background PDF
};

#endif
