/*
 * Project: RooFit
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooFitCore_RooAddHelpers_h
#define RooFit_RooFitCore_RooAddHelpers_h

#include <RooAbsCacheElement.h>
#include <RooArgList.h>
#include <RooAbsReal.h>

class RooAbsPdf;
class RooArgSet;

class AddCacheElem : public RooAbsCacheElement {
public:
   AddCacheElem(RooAbsPdf const &addPdf, RooArgList const &pdfList, RooArgList const &coefList, const RooArgSet *nset,
                const RooArgSet *iset, const char *rangeName, bool projectCoefs, RooArgSet const &refCoefNorm,
                TNamed const *refCoefRangeName, int verboseEval);

   RooArgList containedArgs(Action) override;

   inline double suppNormVal(std::size_t idx) const { return _suppNormList[idx] ? _suppNormList[idx]->getVal() : 1.0; }

   inline bool doProjection() const { return !_projList.empty(); }

   inline double projVal(std::size_t idx) const { return _projList[idx] ? _projList[idx]->getVal() : 1.0; }

   inline double projSuppNormVal(std::size_t idx) const
   {
      return _suppProjList[idx] ? _suppProjList[idx]->getVal() : 1.0;
   }

   inline double rangeProjScaleFactor(std::size_t idx) const { return rangeProjVal(idx) / refRangeProjVal(idx); }

private:
   inline double refRangeProjVal(std::size_t idx) const
   {
      return _refRangeProjList[idx] ? _refRangeProjList[idx]->getVal() : 1.0;
   }

   inline double rangeProjVal(std::size_t idx) const
   {
      return _rangeProjList[idx] ? _rangeProjList[idx]->getVal() : 1.0;
   }

   using OwningArgVector = std::vector<std::unique_ptr<RooAbsReal>>;

   OwningArgVector _suppNormList; ///< Supplemental normalization list
   OwningArgVector _projList;     ///< Projection integrals to be multiplied with coefficients
   OwningArgVector _suppProjList; ///< Projection integrals to multiply with coefficients for supplemental normalization
   OwningArgVector _refRangeProjList; ///< Range integrals to be multiplied with coefficients (reference range)
   OwningArgVector _rangeProjList;    ///< Range integrals to be multiplied with coefficients (target range)
};

class RooAddHelpers {
public:
   static void updateCoefficients(RooAbsPdf const &addPdf, RooArgList const &pdfList, RooArgList const &coefList,
                                  AddCacheElem &cache, const RooArgSet *nset, bool projectCoefs,
                                  RooArgSet const &refCoefNorm, bool allExtendable, std::vector<double> &coefCache,
                                  int &coefErrCount);
};

#endif
