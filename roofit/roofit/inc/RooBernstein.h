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
#include "RooRealProxy.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooBernstein : public RooAbsPdf {
public:

  RooBernstein() ;
  RooBernstein(const char *name, const char *title,
               RooAbsReal& _x, const RooArgList& _coefList) ;

  RooBernstein(const RooBernstein& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooBernstein(*this, newname); }
  inline virtual ~RooBernstein() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

private:

  RooRealProxy _x;
  RooListProxy _coefList ;

  Double_t evaluate() const;
  RooSpan<double> evaluateBatch(std::size_t begin, std::size_t batchSize) const;


  ClassDef(RooBernstein,1) // Bernstein polynomial PDF
};

#endif
