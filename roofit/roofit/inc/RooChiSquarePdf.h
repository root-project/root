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

  RooChiSquarePdf(const RooChiSquarePdf& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooChiSquarePdf(*this, newname); }
  inline virtual ~RooChiSquarePdf() { }

  
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;
  

private:

  RooRealProxy _x;
  RooRealProxy _ndof;

  double evaluate() const;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const;
  inline bool canComputeBatchWithCuda() const { return true; }

  ClassDef(RooChiSquarePdf,1) // Chi Square distribution (eg. the PDF )
};

#endif
