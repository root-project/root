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

  virtual TObject* clone(const char* newname) const { return new RooEffProd(*this,newname); }

  virtual RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype,
                                       const RooArgSet* auxProto, Bool_t verbose) const;

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;

  RooRealProxy _pdf ;               ///< Probability Density function
  RooRealProxy _eff;                ///< Efficiency function

  ClassDef(RooEffProd,2) // Product operator p.d.f of (PDF x efficiency) implementing optimized generator context
};

#endif
