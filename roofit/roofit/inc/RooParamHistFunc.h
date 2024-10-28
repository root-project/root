/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROO_PARAM_HIST_FUNC
#define ROO_PARAM_HIST_FUNC

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooListProxy.h"
#include "RooSetProxy.h"
#include "RooDataHist.h"
#include <list>

class RooParamHistFunc : public RooAbsReal {
public:
  RooParamHistFunc() {} ;
  RooParamHistFunc(const char *name, const char *title, RooDataHist &dh, const RooAbsArg &x,
                   const RooParamHistFunc *paramSource = nullptr, bool paramRelative = true);
  RooParamHistFunc(const RooParamHistFunc& other, const char* name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooParamHistFunc(*this,newname); }

  std::list<double>* binBoundaries(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const override ;
  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override ;
  bool isBinnedDistribution(const RooArgSet&) const override { return true ; }


  bool forceAnalyticalInt(const RooAbsArg&) const override { return true ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet,const char* rangeName=nullptr) const override ;
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=nullptr) const override ;

  double getActual(Int_t ibin) ;
  void setActual(Int_t ibin, double newVal) ;
  double getNominal(Int_t ibin) const ;
  double getNominalError(Int_t ibin) const ;

  const RooArgList& xList() const { return _x ; }
  const RooArgList& paramList() const { return _p ; }
  const RooDataHist& dataHist() const { return _dh ; }
  bool relParam() const { return _relParam; }

 protected:

  friend class RooHistConstraint ;

  RooListProxy  _x ;
  RooListProxy _p ;
  RooDataHist _dh ;
  bool _relParam ;

  double evaluate() const override ;

private:

  ClassDefOverride(RooParamHistFunc,1);
};

#endif
