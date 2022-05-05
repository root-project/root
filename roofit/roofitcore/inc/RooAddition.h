/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAddition.h,v 1.3 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ADDITION
#define ROO_ADDITION

#include "RooAbsReal.h"
#include "RooListProxy.h"
#include "RooObjCacheManager.h"
#include <list>

class RooRealVar;
class RooArgList ;

class RooAddition : public RooAbsReal {
public:

  RooAddition() ;
  RooAddition(const char *name, const char *title, const RooArgList& sumSet, bool takeOwnerShip=false) ;
  RooAddition(const char *name, const char *title, const RooArgList& sumSet1, const RooArgList& sumSet2, bool takeOwnerShip=false) ;
  ~RooAddition() override ;

  RooAddition(const RooAddition& other, const char* name = 0);
  TObject* clone(const char* newname) const override { return new RooAddition(*this, newname); }

  Double_t defaultErrorLevel() const override ;

  void printMetaArgs(std::ostream& os) const override ;

  const RooArgList& list1() const { return _set ; }
  const RooArgList& list() const { return _set ; }

  bool forceAnalyticalInt(const RooAbsArg& /*dep*/) const override {
      // Force RooRealIntegral to offer all observables for internal integration
      return true ;
  }
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& numVars, const char* rangeName=0) const override;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

  bool setData(RooAbsData& data, bool cloneData=true) override ;

  std::list<Double_t>* binBoundaries(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override ;
  std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override ;
  bool isBinnedDistribution(const RooArgSet& obs) const override  ;

  void enableOffsetting(bool) override ;

protected:

  RooArgList   _ownedList ;      ///< List of owned components
  RooListProxy _set ;            ///< set of terms to be summed

  class CacheElem : public RooAbsCacheElement {
  public:
      ~CacheElem() override;
      // Payload
      RooArgList _I ;
      RooArgList containedArgs(Action) override ;
  };
  mutable RooObjCacheManager _cacheMgr ; ///<! The cache manager

  Double_t evaluate() const override;

  ClassDefOverride(RooAddition,3) // Sum of RooAbsReal objects
};

#endif
