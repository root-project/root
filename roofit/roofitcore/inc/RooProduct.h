/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProduct.h,v 1.5 2007/05/11 09:11:30 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   GR, Gerhard Raven,   VU Amsterdan,     graven@nikhef.nl                 *
 *                                                                           *
 * Copyright (c) 2000-2007, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_PRODUCT
#define ROO_PRODUCT

#include "RooAbsReal.h"
#include "RooListProxy.h"
#include "RooObjCacheManager.h"

#include <list>

class RooRealVar;
class RooArgList;


class RooProduct : public RooAbsReal {
public:

  RooProduct() ;
  RooProduct(const char *name, const char *title, const RooArgList& prodSet) ;
  RooProduct(const char *name, const char *title, RooAbsReal& real1, RooAbsReal& real2) ;

  RooProduct(const RooProduct& other, const char* name = 0);

  void addTerm(RooAbsArg* term);

  TObject* clone(const char* newname) const override { return new RooProduct(*this, newname); }
  bool forceAnalyticalInt(const RooAbsArg& dep) const override ;
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
                                                   const RooArgSet* normSet,
                                                   const char* rangeName=nullptr) const override ;
  double analyticalIntegral(Int_t code, const char* rangeName=nullptr) const override;


  RooArgList components() { RooArgList tmp(_compRSet) ; tmp.add(_compCSet) ; return tmp ; }

  ~RooProduct() override ;

  class ProdMap ;

  void printMetaArgs(std::ostream& os) const override ;

  std::list<double>* binBoundaries(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const override ;
  std::list<double>* plotSamplingHint(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const override ;
  bool isBinnedDistribution(const RooArgSet& obs) const override ;

  CacheMode canNodeBeCached() const override { return RooAbsArg::NotAdvised ; } ;
  void setCacheAndTrackHints(RooArgSet&) override ;

protected:

  void ioStreamerPass2() override ;

  RooListProxy _compRSet ;
  RooListProxy _compCSet ;

  class CacheElem : public RooAbsCacheElement {
  public:
      ~CacheElem() override;
      // Payload
      RooArgList _prodList ;
      RooArgList _ownedList ;
      RooArgList containedArgs(Action) override ;
  };
  mutable RooObjCacheManager _cacheMgr ; //! The cache manager


  double calculate(const RooArgList& partIntList) const;
  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const&) const override;

  const char* makeFPName(const char *pfx,const RooArgSet& terms) const ;
  ProdMap* groupProductTerms(const RooArgSet&) const;
  Int_t getPartIntList(const RooArgSet* iset, const char *rangeName=nullptr) const;

  ClassDefOverride(RooProduct,3) // Product of RooAbsReal and/or RooAbsCategory terms
};

#endif
