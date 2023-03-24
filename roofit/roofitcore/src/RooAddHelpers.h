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
                const RooArgSet *iset, RooArgSet const &refCoefNormSet, std::string const &refCoefNormRange);

   RooArgList containedArgs(Action) override;

   inline double suppNormVal(std::size_t idx) const
   {
      return _list[idx].suppNorm ? _list[idx].suppNorm->getVal() : 1.0;
   }

   inline bool doProjection() const { return _doProjection; }

   inline double projVal(std::size_t idx) const { return _list[idx].proj ? _list[idx].proj->getVal() : 1.0; }

   inline double projSuppNormVal(std::size_t idx) const
   {
      return _list[idx].suppProj ? _list[idx].suppProj->getVal() : 1.0;
   }

   inline double rangeProjScaleFactor(std::size_t idx) const
   {
      return _list[idx].rangeProj ? _list[idx].rangeProj->getVal() : 1.0;
   }

private:
   struct Item {
      std::unique_ptr<RooAbsReal> suppNorm; ///< Supplemental normalization
      std::unique_ptr<RooAbsReal> proj;     ///< Projection integral to be multiplied with coefficient
      std::unique_ptr<RooAbsReal>
         suppProj; ///< Projection integral to multiply with coefficient for supplemental normalization
      std::unique_ptr<RooAbsReal>
         rangeProj; ///< Range integral to be multiplied with coefficient (reference to target range)
   };

   void processPdf(RooAbsPdf const &addPdfName, const RooAbsPdf *pdf, const RooAbsReal *coef,
                   RooArgSet const &fullDepList, RooArgSet const *nset, RooArgSet const &nset2,
                   std::string const &normRange, RooArgSet const &refCoefNormSet, std::string const &refCoefNormRange);

   std::vector<Item> _list;
   bool _doProjection = false;
};

class RooAddHelpers {
public:
   static void updateCoefficients(RooAbsPdf const &addPdf, std::vector<double> &coefCache, RooArgList const &pdfList,
                                  bool haveLastCoef, AddCacheElem &cache, const RooArgSet *nset,
                                  RooArgSet const &refCoefNormSet, bool allExtendable, int &coefErrCount);
};

#endif
