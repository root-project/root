/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooEffProd.rdl,v 1.1 2005/06/20 15:44:51 wverkerke Exp $
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

class RooEffProd: public RooAbsPdf {
public:
  // Constructors, assignment etc
  inline RooEffProd() { };
  virtual ~RooEffProd();
  RooEffProd(const char *name, const char *title, RooAbsPdf& pdf, RooAbsReal& efficiency);
  RooEffProd(const RooEffProd& other, const char* name=0);

  virtual TObject* clone(const char* newname) const { return new RooEffProd(*this,newname); }

  virtual RooAbsGenContext* genContext(const RooArgSet &vars, const RooDataSet *prototype,
                                       const RooArgSet* auxProto, Bool_t verbose) const;
protected:
  const RooAbsPdf* pdf() const { const RooAbsPdf* p = dynamic_cast<const RooAbsPdf*>(&_pdf.arg()); assert(p!=0); return p; }
  const RooAbsReal* eff() const { const RooAbsReal* a = dynamic_cast<const RooAbsReal*>( &_eff.arg()); assert(a!=0); return a;}

  // Function evaluation
  virtual Double_t evaluate() const ;

  // the real stuff...
  RooRealProxy _pdf ;     // pdf
  RooRealProxy _eff;      // efficiency

  ClassDef(RooEffProd,1) // Product of PDF with efficiency function with optimized generator context
};

#endif
