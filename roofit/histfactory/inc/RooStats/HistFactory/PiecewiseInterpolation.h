/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: PiecewiseInterpolation.h,v 1.3 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_PIECEWISEINTERPOLATION
#define ROO_PIECEWISEINTERPOLATION

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

#include "RooObjCacheManager.h"
#include <vector>
#include <list>

class RooRealVar;
class RooArgList;

class PiecewiseInterpolation : public RooAbsReal {
public:

  PiecewiseInterpolation() ;
  PiecewiseInterpolation(const char *name, const char *title, const RooAbsReal& nominal, const RooArgList& lowSet, const RooArgList& highSet, const RooArgList& paramSet, bool takeOwnerShip=false) ;
  ~PiecewiseInterpolation() override ;

  PiecewiseInterpolation(const PiecewiseInterpolation& other, const char* name = 0);
  TObject* clone(const char* newname) const override { return new PiecewiseInterpolation(*this, newname); }

  /// Return pointer to the nominal hist function.
  const RooAbsReal* nominalHist() const {
    return &_nominal.arg();
  }

  //  virtual double defaultErrorLevel() const ;

  //  void printMetaArgs(std::ostream& os) const ;

  const RooArgList& lowList() const { return _lowSet ; }
  const RooArgList& highList() const { return _highSet ; }
  const RooArgList& paramList() const { return _paramSet ; }
  const std::vector<int>&  interpolationCodes() const { return _interpCode; }

  //virtual bool forceAnalyticalInt(const RooAbsArg&) const { return true ; }
  bool setBinIntegrator(RooArgSet& allVars) ;

  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet,const char* rangeName=0) const override ;
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const override ;

  void setPositiveDefinite(bool flag=true){_positiveDefinite=flag;}

  void setInterpCode(RooAbsReal& param, int code, bool silent=false);
  void setAllInterpCodes(int code);
  void printAllInterpCodes();

  std::list<double>* binBoundaries(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const override ;
  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override ;
  bool isBinnedDistribution(const RooArgSet& obs) const override ;

protected:

  class CacheElem : public RooAbsCacheElement {
  public:
    CacheElem()  {} ;
    ~CacheElem() override {} ;
    RooArgList containedArgs(Action) override {
      RooArgList ret(_funcIntList) ;
      ret.add(_lowIntList);
      ret.add(_highIntList);
      return ret ;
    }
    RooArgList _funcIntList ;
    RooArgList _lowIntList ;
    RooArgList _highIntList ;
    // will want std::vector<RooRealVar*> for low and high also
  } ;
  mutable RooObjCacheManager _normIntMgr ; ///<! The integration cache manager

  RooRealProxy _nominal;           ///< The nominal value
  RooArgList   _ownedList ;        ///< List of owned components
  RooListProxy _lowSet ;           ///< Low-side variation
  RooListProxy _highSet ;          ///< High-side variation
  RooListProxy _paramSet ;         ///< interpolation parameters
  RooListProxy _normSet ;          ///< interpolation parameters
  bool _positiveDefinite;        ///< protect against negative and 0 bins.

  std::vector<int> _interpCode;

  double evaluate() const override;
  void computeBatch(cudaStream_t*, double* output, size_t size, RooFit::Detail::DataMap const&) const override;

  ClassDefOverride(PiecewiseInterpolation,4) // Sum of RooAbsReal objects
};

#endif
