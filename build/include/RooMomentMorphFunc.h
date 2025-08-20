/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROO_MOMENT_MORPH_FUNC
#define ROO_MOMENT_MORPH_FUNC

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooSetProxy.h"
#include "RooListProxy.h"
#include "RooArgList.h"

#include "TMatrixD.h"
#include "TVectorD.h"

#include <list>

class RooChangeTracker;

class RooMomentMorphFunc : public RooAbsReal {
public:
   enum Setting { Linear, NonLinear, NonLinearPosFractions, NonLinearLinFractions, SineLinear };

   RooMomentMorphFunc();

   RooMomentMorphFunc(const char *name, const char *title, RooAbsReal &_m, const RooArgList &varList,
                      const RooArgList &pdfList, const RooArgList &mrefList, Setting setting = NonLinearPosFractions);
   RooMomentMorphFunc(const char *name, const char *title, RooAbsReal &_m, const RooArgList &varList,
                      const RooArgList &pdfList, const TVectorD &mrefpoints, Setting setting = NonLinearPosFractions);
   RooMomentMorphFunc(const RooMomentMorphFunc &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooMomentMorphFunc(*this, newname); }
   ~RooMomentMorphFunc() override;

   void setMode(const Setting &setting) { _setting = setting; }

   void useHorizontalMorphing(bool val) { _useHorizMorph = val; }

   virtual bool selfNormalized() const
   {
      // P.d.f is self normalized
      return true;
   }

   double getValV(const RooArgSet *set = nullptr) const override;
   RooAbsReal *sumFunc(const RooArgSet *nset);
   const RooAbsReal *sumFunc(const RooArgSet *nset) const;

   std::list<double> *plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const override;
   std::list<double> *binBoundaries(RooAbsRealLValue & /*obs*/, double /*xlo*/, double /*xhi*/) const override;
   bool isBinnedDistribution(const RooArgSet &obs) const override;

protected:
   class CacheElem : public RooAbsCacheElement {
   public:
      CacheElem(RooAbsReal &sumFunc, RooChangeTracker &tracker, const RooArgList &flist)
         : _sumFunc(&sumFunc), _tracker(&tracker)
      {
         _frac.add(flist);
      };
      ~CacheElem() override;
      RooArgList containedArgs(Action) override;
      RooAbsReal *_sumFunc;
      RooChangeTracker *_tracker;
      RooArgList _frac;

      RooRealVar *frac(Int_t i);
      const RooRealVar *frac(Int_t i) const;
      void calculateFractions(const RooMomentMorphFunc &self, bool verbose = true) const;
   };
   mutable RooObjCacheManager _cacheMgr; //! The cache manager
   mutable RooArgSet *_curNormSet = nullptr; //! Current normalization set

   friend class CacheElem; // Cache needs to be able to clear _norm pointer

   double evaluate() const override;

   void initialize();
   CacheElem *getCache(const RooArgSet *nset) const;

   inline Int_t ij(const Int_t &i, const Int_t &j) const { return (i * _varList.size() + j); }
   int idxmin(const double &m) const;
   int idxmax(const double &m) const;

   RooRealProxy m;
   RooSetProxy _varList;
   RooListProxy _pdfList;
   mutable TVectorD *_mref = nullptr;

   mutable TMatrixD *_M = nullptr;

   Setting _setting;

   bool _useHorizMorph = true;

   ClassDefOverride(RooMomentMorphFunc, 3);
};

#endif
