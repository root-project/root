/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooEffProd.h,v 1.2 2007/05/11 10:14:56 verkerke Exp $
 * Authors:                                                                  *
 *   GR, Gerhard Raven, NIKHEF/VU                                            *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_EFF_PROD
#define ROO_EFF_PROD

#include "RooAbsPdf.h"
#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooObjCacheManager.h"

class RooEffProd: public RooAbsPdf {
public:
  // Constructors, assignment etc
  RooEffProd(const char *name, const char *title, RooAbsPdf& pdf, RooAbsReal& efficiency);
  RooEffProd(const RooEffProd& other, const char* name=0);

  TObject* clone(const char* newname) const override { return new RooEffProd(*this,newname); }

  RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype,
                                       const RooArgSet* auxProto, bool verbose) const override;

protected:

  // Function evaluation
  Double_t evaluate() const override ;

  RooRealProxy _pdf ;               ///< Probability Density function
  RooRealProxy _eff;                ///< Efficiency function

  ClassDefOverride(RooEffProd,2) // Product operator p.d.f of (PDF x efficiency) implementing optimized generator context
};

#endif
