/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOMOMENTMORPHFUNCND
#define ROOMOMENTMORPHFUNCND

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooSetProxy.h"
#include "RooListProxy.h"
#include "RooArgList.h"
#include "RooBinning.h"

#include "TMatrixD.h"
#include "TMap.h"

#include <vector>
#include <map>

class RooChangeTracker;
class RooRealSumFunc;

class RooMomentMorphFuncND : public RooAbsReal {

public:
   using Base_t = RooAbsReal;

   class Grid2 {
   public:
      Grid2(){};
      Grid2(const Grid2 &other);
      Grid2(const RooAbsBinning &binning_x) { _grid.push_back(binning_x.clone()); };
      Grid2(const RooAbsBinning &binning_x, const RooAbsBinning &binning_y)
      {
         _grid.push_back(binning_x.clone());
         _grid.push_back(binning_y.clone());
      };
      Grid2(const RooAbsBinning &binning_x, const RooAbsBinning &binning_y, const RooAbsBinning &binning_z)
      {
         _grid.push_back(binning_x.clone());
         _grid.push_back(binning_y.clone());
         _grid.push_back(binning_z.clone());
      };
      Grid2(std::vector<RooAbsBinning *> const &binnings)
      {
         for (unsigned int i = 0; i < binnings.size(); i++) {
            _grid.push_back(binnings[i]->clone());
         }
      };

      virtual ~Grid2();

      void addPdf(const RooAbsReal &func, int bin_x);
      void addPdf(const RooAbsReal &func, int bin_x, int bin_y);
      void addPdf(const RooAbsReal &func, int bin_x, int bin_y, int bin_z);
      void addPdf(const RooAbsReal &func, std::vector<int> bins);
      void addBinning(const RooAbsBinning &binning) { _grid.push_back(binning.clone()); };

      mutable std::vector<RooAbsBinning *> _grid;
      mutable RooArgList _pdfList;
      mutable std::map<std::vector<int>, int> _pdfMap;

      mutable std::vector<std::vector<double>> _nref;
      mutable std::vector<int> _nnuis;

      ClassDef(RooMomentMorphFuncND::Grid2, 1);
   };

   using Grid = Grid2;

protected:
   class CacheElem : public RooAbsCacheElement {
   public:
      CacheElem(std::unique_ptr<RooAbsReal> && sumFunc, std::unique_ptr<RooChangeTracker> && tracker, const RooArgList &flist);
      ~CacheElem() override;
      RooArgList containedArgs(Action) override;
      std::unique_ptr<RooAbsReal> _sum;
      std::unique_ptr<RooChangeTracker> _tracker;
      RooArgList _frac;

      RooRealVar *frac(int i);
      const RooRealVar *frac(int i) const;
      void calculateFractions(const RooMomentMorphFuncND &self, bool verbose = true) const;
   };

public:
   enum Setting { Linear, SineLinear, NonLinear, NonLinearPosFractions, NonLinearLinFractions };

   RooMomentMorphFuncND();
   RooMomentMorphFuncND(const char *name, const char *title, RooAbsReal &_m, const RooArgList &varList,
                        const RooArgList &pdfList, const RooArgList &mrefList, Setting setting);
   RooMomentMorphFuncND(const char *name, const char *title, const RooArgList &parList, const RooArgList &obsList,
                        const Grid2 &referenceGrid, Setting setting);
   RooMomentMorphFuncND(const RooMomentMorphFuncND &other, const char *name = nullptr);
   RooMomentMorphFuncND(const char *name, const char *title, RooAbsReal &_m, const RooArgList &varList,
                        const RooArgList &pdfList, const TVectorD &mrefpoints, Setting setting);
   ~RooMomentMorphFuncND() override;
   TObject *clone(const char *newname) const override { return new RooMomentMorphFuncND(*this, newname); }

   void setMode(const Setting &setting) { _setting = setting; }
   /// Setting flag makes this RooMomentMorphFuncND instance behave like the
   /// former RooMomentMorphND class, with the the only difference being the
   /// base class. If you want to create a pdf object that behaves exactly like
   /// the old RooMomentMorphND, you can do it as follows:
   ///
   /// ```C++
   /// RooMomentMorphFuncND func{<c'tor args you previously passed to RooMomentMorphFunc>};
   ///
   /// func.setPdfMode(); // change behavior to be exactly like the former RooMomentMorphND
   ///
   /// // Pass the selfNormalized=true` flag to the wrapper because the
   /// RooMomentMorphFuncND already normalizes itself in pdf mode.
   /// RooWrapperPdf pdf{"pdf_name", "pdf_name", func, /*selfNormalized=*/true};
   /// ```
   void setPdfMode(bool flag=true) { _isPdfMode = flag; }
   bool setBinIntegrator(RooArgSet &allVars);
   void useHorizontalMorphing(bool val) { _useHorizMorph = val; }

   double evaluate() const override;
   double getValV(const RooArgSet *set = nullptr) const override;

protected:
   void initialize();

   RooAbsReal *sumFunc(const RooArgSet *nset);
   CacheElem *getCache(const RooArgSet *nset) const;

   void findShape(const std::vector<double> &x) const;

   friend class CacheElem;
   friend class Grid2;

   mutable RooObjCacheManager _cacheMgr;     ///<! Transient cache manager
   mutable RooArgSet *_curNormSet = nullptr; ///<! Transient cache manager

   RooListProxy _parList;
   RooSetProxy _obsList;
   mutable Grid2 _referenceGrid;
   RooListProxy _pdfList;

   mutable std::unique_ptr<TMatrixD> _M;
   mutable std::unique_ptr<TMatrixD> _MSqr;
   mutable std::vector<std::vector<double>> _squareVec;
   mutable std::vector<int> _squareIdx;

   Setting _setting;
   bool _useHorizMorph;
   bool _isPdfMode = false;

   inline int sij(const int &i, const int &j) const { return (i * _obsList.size() + j); }

   ClassDefOverride(RooMomentMorphFuncND, 4);
};

#endif
