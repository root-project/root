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
#ifndef ROO_BERNSTEIN
#define ROO_BERNSTEIN

#include "RooAbsPdf.h"
#include "RooTemplateProxy.h"
#include "RooRealVar.h"
#include "RooListProxy.h"
#include "RooAbsRealLValue.h"
#include <string>

class RooRealVar;
class RooArgList;

class RooBernstein : public RooAbsPdf {
public:

  RooBernstein() {}
  RooBernstein(const char *name, const char *title,
               RooAbsRealLValue& _x, const RooArgList& _coefList) ;

  RooBernstein(const RooBernstein &other, const char *name = nullptr);

  TObject* clone(const char* newname) const override { return new RooBernstein(*this, newname); }
  inline ~RooBernstein() override { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override ;
  void selectNormalizationRange(const char* rangeName=nullptr, bool force=false) override ;

private:

  RooTemplateProxy<RooAbsRealLValue> _x ;
  RooListProxy _coefList ;
  std::string _refRangeName ;

  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;
  inline bool canComputeBatchWithCuda() const override { return true; }

  ClassDefOverride(RooBernstein,2) // Bernstein polynomial PDF
};

#endif
