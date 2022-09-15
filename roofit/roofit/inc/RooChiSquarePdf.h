/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id$
 * Authors:                                                                  *
 *   Kyle Cranmer
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_CHISQUARE
#define ROO_CHISQUARE

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooRealVar;
class RooArgList ;

class RooChiSquarePdf : public RooAbsPdf {
public:

  RooChiSquarePdf() ;
  RooChiSquarePdf(const char *name, const char *title,
               RooAbsReal& x,  RooAbsReal& ndof) ;

  RooChiSquarePdf(const RooChiSquarePdf& other, const char *name = nullptr);
  TObject* clone(const char* newname) const override { return new RooChiSquarePdf(*this, newname); }
  inline ~RooChiSquarePdf() override { }


  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;


private:

  RooRealProxy _x;
  RooRealProxy _ndof;

  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

  ClassDefOverride(RooChiSquarePdf,1) // Chi Square distribution (eg. the PDF )
};

#endif
