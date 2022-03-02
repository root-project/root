/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealSumFunc.h,v 1.10 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_REAL_SUM_FUNC
#define ROO_REAL_SUM_FUNC

#include "RooAbsPdf.h"
#include "RooListProxy.h"
#include "RooAICRegistry.h"
#include "RooObjCacheManager.h"
#include <list>

class RooRealSumFunc : public RooAbsReal {
public:
   RooRealSumFunc();
   RooRealSumFunc(const char *name, const char *title);
   RooRealSumFunc(const char *name, const char *title, const RooArgList &funcList, const RooArgList &coefList);
   RooRealSumFunc(const char *name, const char *title, RooAbsReal &func1, RooAbsReal &func2, RooAbsReal &coef1);
   RooRealSumFunc(const RooRealSumFunc &other, const char *name = 0);
   TObject *clone(const char *newname) const override { return new RooRealSumFunc(*this, newname); }
   ~RooRealSumFunc() override;

   Double_t evaluate() const override;
   Bool_t checkObservables(const RooArgSet *nset) const override;

   void computeBatch(cudaStream_t*, double* output, size_t size, RooBatchCompute::DataMap&) const override;

   Bool_t forceAnalyticalInt(const RooAbsArg &arg) const override { return arg.isFundamental(); }
   Int_t getAnalyticalIntegralWN(RooArgSet &allVars, RooArgSet &numVars, const RooArgSet *normSet,
                                 const char *rangeName = 0) const override;
   Double_t analyticalIntegralWN(Int_t code, const RooArgSet *normSet, const char *rangeName = 0) const override;

   const RooArgList &funcList() const { return _funcList; }
   const RooArgList &coefList() const { return _coefList; }

   void printMetaArgs(std::ostream &os) const override;

   std::list<Double_t> *binBoundaries(RooAbsRealLValue & /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override;
   std::list<Double_t> *plotSamplingHint(RooAbsRealLValue & /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override;
   Bool_t isBinnedDistribution(const RooArgSet &obs) const override;

   void setFloor(Bool_t flag) { _doFloor = flag; }
   Bool_t getFloor() const { return _doFloor; }
   static void setFloorGlobal(Bool_t flag) { _doFloorGlobal = flag; }
   static Bool_t getFloorGlobal() { return _doFloorGlobal; }

   CacheMode canNodeBeCached() const override { return RooAbsArg::NotAdvised; };
   void setCacheAndTrackHints(RooArgSet &) override;

protected:
   class CacheElem : public RooAbsCacheElement {
   public:
      CacheElem(){};
      ~CacheElem() override{};
      RooArgList containedArgs(Action) override
      {
         RooArgList ret(_funcIntList);
         ret.add(_funcNormList);
         return ret;
      }
      RooArgList _funcIntList;
      RooArgList _funcNormList;
   };
   mutable RooObjCacheManager _normIntMgr; //! The integration cache manager

   Bool_t _haveLastCoef;

   RooListProxy _funcList; ///<  List of component FUNCs
   RooListProxy _coefList; ///<  List of coefficients
   TIterator *_funcIter;   ///<! Iterator over FUNC list
   TIterator *_coefIter;   ///<! Iterator over coefficient list

   Bool_t _doFloor;              ///< Introduce floor at zero in pdf
   static Bool_t _doFloorGlobal; ///< Global flag for introducing floor at zero in pdf

private:
   ClassDefOverride(RooRealSumFunc, 4) // PDF constructed from a sum of (non-pdf) functions
};

#endif
