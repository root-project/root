/*
 * Project: RooFit
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooAddHelpers.h"

#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooNaNPacker.h>
#include <RooRealConstant.h>
#include <RooRealIntegral.h>
#include <RooRealVar.h>
#include <RooRatio.h>

////////////////////////////////////////////////////////////////////////////////
/// Create a RooAddPdf cache element for a given normalization set and
/// projection configuration.

AddCacheElem::AddCacheElem(RooAbsPdf const &addPdf, RooArgList const &pdfList, RooArgList const &coefList,
                           const RooArgSet *nset, const RooArgSet *iset, RooArgSet const &refCoefNormSet,
                           std::string const &refCoefNormRange)
{
   // We put the normRange into a std::string to not have to deal with
   // nullptr vs. "" ambiguities
   const std::string normRange = addPdf.normRange() ? addPdf.normRange() : "";

   _list.reserve(pdfList.size());

   // Retrieve the combined set of dependents of this PDF ;
   RooArgSet fullDepList;
   addPdf.getObservables(nset, fullDepList);
   if (iset) {
      fullDepList.remove(*iset, true, true);
   }

   bool hasPdfWithCustomRange = false;
   for (std::size_t i = 0; i < pdfList.size(); ++i) {
      auto pdf = static_cast<const RooAbsPdf *>(pdfList.at(i));
      hasPdfWithCustomRange |= pdf->normRange() != nullptr;
   }

   const bool projectCoefsForRangeReasons = !refCoefNormRange.empty() || !normRange.empty() || hasPdfWithCustomRange;

   bool requiresProjection = !refCoefNormSet.empty() || projectCoefsForRangeReasons;

   // Reduce iset/nset to actual dependents of this PDF
   RooArgSet nset2;
   if (requiresProjection) {
      if (nset)
         addPdf.getObservables(nset, nset2);
      oocxcoutD(&addPdf, Caching) << addPdf.ClassName() << "(" << addPdf.GetName()
                                  << ")::getPC nset = " << (nset ? *nset : RooArgSet()) << " nset2 = " << nset2
                                  << std::endl;

      if (nset2.empty() && !refCoefNormSet.empty()) {
         // cout << "WVE: evaluating RooAddPdf without normalization, but have reference normalization for coefficient
         // definition" << std::endl ;

         nset2.add(refCoefNormSet);
      }
   }

   _doProjection = !nset2.equals(refCoefNormSet) || projectCoefsForRangeReasons;

   // Fill with dummy unit RRVs for now
   for (std::size_t i = 0; i < pdfList.size(); ++i) {
      auto pdf = static_cast<const RooAbsPdf *>(pdfList.at(i));
      auto coef = static_cast<const RooAbsReal *>(coefList.at(i));
      processPdf(addPdf, pdf, coef, fullDepList, nset, nset2, normRange, refCoefNormSet, refCoefNormRange);
   }
}

void AddCacheElem::processPdf(RooAbsPdf const &addPdf, const RooAbsPdf *pdf, const RooAbsReal *coef,
                              RooArgSet const &fullDepList, RooArgSet const *nset, RooArgSet const &nset2,
                              std::string const &normRange, RooArgSet const &refCoefNormSet,
                              std::string const &refCoefNormRange)
{
   std::string addPdfName = addPdf.GetName();

   auto &item = _list.emplace_back();

   // *** PART 1 : Create supplemental normalization list ***

   // Start with full list of dependents
   RooArgSet supNSet(fullDepList);

   // Remove PDF dependents
   if (auto pdfDeps = std::unique_ptr<RooArgSet>{pdf->getObservables(nset)}) {
      supNSet.remove(*pdfDeps, true, true);
   }

   // Remove coef dependents
   if (auto coefDeps = std::unique_ptr<RooArgSet>{coef ? coef->getObservables(nset) : nullptr}) {
      supNSet.remove(*coefDeps, true, true);
   }

   std::unique_ptr<RooAbsReal> snorm;
   auto name = addPdfName + "_" + pdf->GetName() + "_SupNorm";
   if (!supNSet.empty()) {
      snorm = std::make_unique<RooRealIntegral>(name.c_str(), "Supplemental normalization integral",
                                                RooRealConstant::value(1.0), supNSet);
   }

   if (!normRange.empty()) {
      auto snormTerm = std::unique_ptr<RooAbsReal>(pdf->createIntegral(*nset, *nset, normRange.c_str()));
      if (snorm) {
         auto oldSnorm = std::move(snorm);
         snorm = std::make_unique<RooProduct>("snorm", "snorm", *oldSnorm.get(), *snormTerm.get());
         snorm->addOwnedComponents(std::move(snormTerm), std::move(oldSnorm));
      } else {
         snorm = std::move(snormTerm);
      }
   }
   item.suppNorm = std::move(snorm);

   // *** PART 2 : Create projection coefficients ***

   // If no projections required stop here
   if (!_doProjection) {
      return;
   }

   // Recalculate projection integrals of PDFs

   // Calculate projection integral
   std::unique_ptr<RooAbsReal> pdfProj;
   if (!refCoefNormSet.empty() && !nset2.equals(refCoefNormSet)) {
      pdfProj = std::unique_ptr<RooAbsReal>{pdf->createIntegral(nset2, refCoefNormSet, normRange.c_str())};
      pdfProj->setOperMode(addPdf.operMode());
   }

   item.proj = std::move(pdfProj);

   // Calculation optional supplemental normalization term
   RooArgSet supNormSet(refCoefNormSet);
   auto deps = std::unique_ptr<RooArgSet>{pdf->getParameters(RooArgSet())};
   supNormSet.remove(*deps, true, true);

   std::unique_ptr<RooAbsReal> sProjNorm;
   auto sProjNormName = std::string(addPdf.GetName()) + "_" + pdf->GetName() + "_ProjSupNorm";
   if (!supNormSet.empty() && !nset2.equals(refCoefNormSet)) {
      sProjNorm =
         std::make_unique<RooRealIntegral>(sProjNormName.c_str(), "Projection Supplemental normalization integral",
                                           RooRealConstant::value(1.0), supNormSet);
   }
   item.suppProj = std::move(sProjNorm);

   // Calculate range adjusted projection integral
   std::unique_ptr<RooAbsReal> rangeProj2;
   if (normRange != refCoefNormRange) {
      RooArgSet tmp;
      pdf->getObservables(refCoefNormSet.empty() ? nset : &refCoefNormSet, tmp);
      auto int1 = std::unique_ptr<RooAbsReal>{pdf->createIntegral(tmp, tmp, normRange.c_str())};
      auto int2 = std::unique_ptr<RooAbsReal>{pdf->createIntegral(tmp, tmp, refCoefNormRange.c_str())};
      rangeProj2 = std::make_unique<RooRatio>("rangeProj", "rangeProj", *int1, *int2);
      rangeProj2->addOwnedComponents(std::move(int1), std::move(int2));
   }

   item.rangeProj = std::move(rangeProj2);
}

////////////////////////////////////////////////////////////////////////////////
/// List all RooAbsArg derived contents in this cache element

RooArgList AddCacheElem::containedArgs(Action)
{
   RooArgList allNodes;
   // need to iterate manually because _suppProjList can contain nullptr
   for (auto const &item : _list) {
      if (item.proj)
         allNodes.add(*item.proj);
      if (item.suppProj)
         allNodes.add(*item.suppProj);
      if (item.rangeProj)
         allNodes.add(*item.rangeProj);
   }

   return allNodes;
}

////////////////////////////////////////////////////////////////////////////////
/// Update the RooAddPdf coefficients for a given normalization set and
/// projection configuration. The `coefCache` argument should have the same
/// size as `pdfList`. It needs to be initialized with the raw values of the
/// coefficients, as obtained from the `_coefList` proxy in the RooAddPdf. If
/// the last coefficient is not given, the initial value of the last element of
/// `_coefCache` does not matter. After this function, the `_coefCache` will be
/// filled with the correctly scaled coefficients for each pdf.

void RooAddHelpers::updateCoefficients(RooAbsPdf const &addPdf, std::vector<double> &coefCache,
                                       RooArgList const &pdfList, bool haveLastCoef, AddCacheElem &cache,
                                       const RooArgSet *nset, RooArgSet const &refCoefNormSet, bool allExtendable,
                                       int &coefErrCount)
{
   // Straight coefficients
   if (allExtendable) {

      // coef[i] = expectedEvents[i] / SUM(expectedEvents)
      double coefSum(0);
      std::size_t i = 0;
      for (auto arg : pdfList) {
         auto pdf = static_cast<RooAbsPdf *>(arg);
         coefCache[i] = pdf->expectedEvents(!refCoefNormSet.empty() ? &refCoefNormSet : nset);
         coefSum += coefCache[i];
         i++;
      }

      if (coefSum == 0.) {
         oocoutW(&addPdf, Eval) << addPdf.ClassName() << "::updateCoefCache(" << addPdf.GetName()
                                << ") WARNING: total number of expected events is 0" << std::endl;
      } else {
         for (std::size_t j = 0; j < pdfList.size(); j++) {
            coefCache[j] /= coefSum;
         }
      }

   } else {
      if (haveLastCoef) {

         // coef[i] = coef[i] / SUM(coef)
         double coefSum = std::accumulate(coefCache.begin(), coefCache.end(), 0.0);
         if (coefSum == 0.) {
            oocoutW(&addPdf, Eval) << addPdf.ClassName() << "::updateCoefCache(" << addPdf.GetName()
                                   << ") WARNING: sum of coefficients is zero 0" << std::endl;
         } else {
            const double invCoefSum = 1. / coefSum;
            for (std::size_t j = 0; j < coefCache.size(); j++) {
               coefCache[j] *= invCoefSum;
            }
         }
      } else {

         // coef[i] = coef[i] ; coef[n] = 1-SUM(coef[0...n-1])
         double lastCoef = 1.0 - std::accumulate(coefCache.begin(), coefCache.end() - 1, 0.0);
         coefCache.back() = lastCoef;

         // Treat coefficient degeneration
         const float coefDegen = lastCoef < 0. ? -lastCoef : (lastCoef > 1. ? lastCoef - 1. : 0.);
         if (coefDegen > 1.E-5) {
            coefCache.back() = RooNaNPacker::packFloatIntoNaN(100.f * coefDegen);

            std::stringstream msg;
            if (coefErrCount-- > 0) {
               msg << "RooAddPdf::updateCoefCache(" << addPdf.GetName()
                   << " WARNING: sum of PDF coefficients not in range [0-1], value=" << 1 - lastCoef;
               if (coefErrCount == 0) {
                  msg << " (no more will be printed)";
               }
               oocoutW(&addPdf, Eval) << msg.str() << std::endl;
            }
         }
      }
   }

   // Stop here if not projection is required or needed
   if (!cache.doProjection()) {
      return;
   }

   // Adjust coefficients for given projection
   double coefSum(0);
   {
      RooAbsReal::GlobalSelectComponentRAII compRAII(true);

      for (std::size_t i = 0; i < pdfList.size(); i++) {
         coefCache[i] *= cache.projVal(i) / cache.projSuppNormVal(i) * cache.rangeProjScaleFactor(i);
         coefSum += coefCache[i];
      }
   }

   if ((RooMsgService::_debugCount > 0) &&
       RooMsgService::instance().isActive(&addPdf, RooFit::Caching, RooFit::DEBUG)) {
      for (std::size_t i = 0; i < pdfList.size(); ++i) {
         ooccoutD(&addPdf, Caching) << " ALEX:   POST-SYNC coef[" << i << "] = " << coefCache[i]
                                    << " ( _coefCache[i]/coefSum = " << coefCache[i] * coefSum << "/" << coefSum
                                    << " ) " << std::endl;
      }
   }

   if (coefSum == 0.) {
      oocoutE(&addPdf, Eval) << addPdf.ClassName() << "::updateCoefCache(" << addPdf.GetName()
                             << ") sum of coefficients is zero." << std::endl;
   }

   for (std::size_t i = 0; i < pdfList.size(); i++) {
      coefCache[i] /= coefSum;
   }
}
