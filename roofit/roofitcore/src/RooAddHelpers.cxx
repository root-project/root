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

AddCacheElem::AddCacheElem(RooAbsPdf const &addPdf, RooArgList const &pdfList, RooArgList const &coefList,
                           const RooArgSet *nset, const RooArgSet *iset, const char *rangeName, bool projectCoefs,
                           RooArgSet const &refCoefNorm, TNamed const *refCoefRangeName, int verboseEval)
{
   _suppNormList.reserve(pdfList.size());

   // *** PART 1 : Create supplemental normalization list ***

   // Retrieve the combined set of dependents of this PDF ;
   RooArgSet fullDepList;
   addPdf.getObservables(nset, fullDepList);
   if (iset) {
      fullDepList.remove(*iset, true, true);
   }

   // Fill with dummy unit RRVs for now
   for (std::size_t i = 0; i < pdfList.size(); ++i) {
      auto pdf = static_cast<const RooAbsPdf *>(pdfList.at(i));
      auto coef = static_cast<const RooAbsReal *>(coefList.at(i));

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
      auto name = std::string(addPdf.GetName()) + "_" + pdf->GetName() + "_SupNorm";
      if (!supNSet.empty()) {
         snorm = std::make_unique<RooRealIntegral>(name.c_str(), "Supplemental normalization integral",
                                                   RooRealConstant::value(1.0), supNSet);
         oocxcoutD(&addPdf, Caching) << addPdf.ClassName() << " " << addPdf.GetName()
                                     << " making supplemental normalization set " << supNSet << " for pdf component "
                                     << pdf->GetName() << std::endl;
      }
      _suppNormList.emplace_back(std::move(snorm));
   }

   if (verboseEval > 1) {
      oocxcoutD(&addPdf, Caching) << addPdf.ClassName() << "::syncSuppNormList(" << addPdf.GetName()
                                  << ") synching supplemental normalization list for norm"
                                  << (nset ? *nset : RooArgSet()) << std::endl;
   }

   // *** PART 2 : Create projection coefficients ***

   //   cout << " this = " << this << " (" << GetName() << ")" << std::endl ;
   //   cout << "projectCoefs = " << (_projectCoefs?"T":"F") << std::endl ;
   //   cout << "_normRange.Length() = " << _normRange.Length() << std::endl ;

   // If no projections required stop here
   if (!projectCoefs && !rangeName) {
      //     cout << " no projection required" << std::endl ;
      return;
   }

   //   cout << "calculating projection" << std::endl ;

   // Reduce iset/nset to actual dependents of this PDF
   RooArgSet nset2;
   if (nset)
      addPdf.getObservables(nset, nset2);
   oocxcoutD(&addPdf, Caching) << addPdf.ClassName() << "(" << addPdf.GetName()
                               << ")::getPC nset = " << (nset ? *nset : RooArgSet()) << " nset2 = " << nset2
                               << std::endl;

   if (nset2.empty() && !refCoefNorm.empty()) {
      // cout << "WVE: evaluating RooAddPdf without normalization, but have reference normalization for coefficient
      // definition" << std::endl ;

      nset2.add(refCoefNorm);
      if (refCoefRangeName) {
         rangeName = RooNameReg::str(refCoefRangeName);
      }
   }

   // Check if requested transformation is not identity
   if (!nset2.equals(refCoefNorm) || refCoefRangeName != 0 || rangeName != 0 || addPdf.normRange()) {

      oocxcoutD(&addPdf, Caching) << "ALEX:     " << addPdf.ClassName() << "::syncCoefProjList(" << addPdf.GetName()
                                  << ") projecting coefficients from " << nset2 << (rangeName ? ":" : "")
                                  << (rangeName ? rangeName : "") << " to "
                                  << ((!refCoefNorm.empty()) ? refCoefNorm : nset2) << (refCoefRangeName ? ":" : "")
                                  << (refCoefRangeName ? RooNameReg::str(refCoefRangeName) : "") << std::endl;

      // Recalculate projection integrals of PDFs
      for (auto *thePdf : static_range_cast<const RooAbsPdf *>(pdfList)) {

         // Calculate projection integral
         std::unique_ptr<RooAbsReal> pdfProj;
         if (!nset2.equals(refCoefNorm)) {
            pdfProj = std::unique_ptr<RooAbsReal>{thePdf->createIntegral(nset2, refCoefNorm, addPdf.normRange())};
            pdfProj->setOperMode(addPdf.operMode());
            oocxcoutD(&addPdf, Caching) << addPdf.ClassName() << "(" << addPdf.GetName() << ")::getPC nset2(" << nset2
                                        << ")!=_refCoefNorm(" << refCoefNorm << ") --> pdfProj = " << pdfProj->GetName()
                                        << std::endl;
            oocxcoutD(&addPdf, Caching) << " " << addPdf.ClassName() << "::syncCoefProjList(" << addPdf.GetName()
                                        << ") PP = " << pdfProj->GetName() << std::endl;
         }

         _projList.emplace_back(std::move(pdfProj));

         // Calculation optional supplemental normalization term
         RooArgSet supNormSet(refCoefNorm);
         auto deps = std::unique_ptr<RooArgSet>{thePdf->getParameters(RooArgSet())};
         supNormSet.remove(*deps, true, true);

         std::unique_ptr<RooAbsReal> snorm;
         auto name = std::string(addPdf.GetName()) + "_" + thePdf->GetName() + "_ProjSupNorm";
         if (!supNormSet.empty() && !nset2.equals(refCoefNorm)) {
            snorm = std::make_unique<RooRealIntegral>(name.c_str(), "Projection Supplemental normalization integral",
                                                      RooRealConstant::value(1.0), supNormSet);
            oocxcoutD(&addPdf, Caching) << " " << addPdf.ClassName() << "::syncCoefProjList(" << addPdf.GetName()
                                        << ") SN = " << snorm->GetName() << std::endl;
         }
         _suppProjList.emplace_back(std::move(snorm));

         // Calculate reference range adjusted projection integral
         std::unique_ptr<RooAbsReal> rangeProj1;

         //    cout << "ALEX >>>> RooAddPdf(" << GetName() << ")::getPC refCoefRangeName WVE = "
         //       <<(refCoefRangeName?":":"") << (refCoefRangeName?RooNameReg::str(refCoefRangeName):"")
         //       <<" refCoefRangeName AK = "  << (refCoefRangeName?refCoefRangeName->GetName():"")
         //       << " && _refCoefNorm" << _refCoefNorm << " with size = _refCoefNorm.size() " << _refCoefNorm.size() <<
         //       std::endl ;

         // Check if refCoefRangeName is identical to default range for all observables,
         // If so, substitute by unit integral

         // ----------
         RooArgSet tmpObs;
         thePdf->getObservables(&refCoefNorm, tmpObs);
         bool allIdent = true;
         for (auto *rvarg : dynamic_range_cast<RooRealVar *>(tmpObs)) {
            if (rvarg) {
               if (rvarg->getMin(RooNameReg::str(refCoefRangeName)) != rvarg->getMin() ||
                   rvarg->getMax(RooNameReg::str(refCoefRangeName)) != rvarg->getMax()) {
                  allIdent = false;
               }
            }
         }
         // -------------

         if (refCoefRangeName && !refCoefNorm.empty() && !allIdent) {

            RooArgSet tmp;
            thePdf->getObservables(&refCoefNorm, tmp);
            rangeProj1 =
               std::unique_ptr<RooAbsReal>{thePdf->createIntegral(tmp, tmp, RooNameReg::str(refCoefRangeName))};

            // rangeProj1->setOperMode(operMode()) ;
            oocxcoutD(&addPdf, Caching) << " " << addPdf.ClassName() << "::syncCoefProjList(" << addPdf.GetName()
                                        << ") R1 = " << rangeProj1->GetName() << std::endl;
         }
         _refRangeProjList.emplace_back(std::move(rangeProj1));

         // Calculate range adjusted projection integral
         std::unique_ptr<RooAbsReal> rangeProj2;
         oocxcoutD(&addPdf, Caching) << addPdf.ClassName() << "::syncCoefProjList(" << addPdf.GetName()
                                     << ") rangename = " << (rangeName ? rangeName : "<null>")
                                     << " nset = " << (nset ? *nset : RooArgSet()) << std::endl;
         if (rangeName && !refCoefNorm.empty()) {

            rangeProj2 = std::unique_ptr<RooAbsReal>{thePdf->createIntegral(refCoefNorm, refCoefNorm, rangeName)};
            // rangeProj2->setOperMode(operMode()) ;

         } else if (addPdf.normRange()) {

            RooArgSet tmp;
            thePdf->getObservables(&refCoefNorm, tmp);
            rangeProj2 = std::unique_ptr<RooAbsReal>{thePdf->createIntegral(tmp, tmp, addPdf.normRange())};
            oocxcoutD(&addPdf, Caching) << " " << addPdf.ClassName() << "::syncCoefProjList(" << addPdf.GetName()
                                        << ") R2 = " << rangeProj2->GetName() << std::endl;
         }
         _rangeProjList.emplace_back(std::move(rangeProj2));
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// List all RooAbsArg derived contents in this cache element

RooArgList AddCacheElem::containedArgs(Action)
{
   RooArgList allNodes;
   // need to iterate manually because _suppProjList can contain nullptr
   for (auto const &arg : _projList) {
      if (arg)
         allNodes.add(*arg);
   }
   for (auto const &arg : _suppProjList) {
      if (arg)
         allNodes.add(*arg);
   }
   for (auto const &arg : _refRangeProjList) {
      if (arg)
         allNodes.add(*arg);
   }
   for (auto const &arg : _rangeProjList) {
      if (arg)
         allNodes.add(*arg);
   }

   return allNodes;
}

void RooAddHelpers::updateCoefficients(RooAbsPdf const &addPdf, RooArgList const &pdfList, RooArgList const &coefList,
                                       AddCacheElem &cache, const RooArgSet *nset, bool projectCoefs,
                                       RooArgSet const &refCoefNorm, bool allExtendable, std::vector<double> &coefCache,
                                       int &coefErrCount)
{
   bool haveLastCoef = pdfList.size() == coefList.size();

   coefCache.resize(haveLastCoef ? coefList.size() : pdfList.size(), 0.);

   // Straight coefficients
   if (allExtendable) {

      // coef[i] = expectedEvents[i] / SUM(expectedEvents)
      double coefSum(0);
      std::size_t i = 0;
      for (auto arg : pdfList) {
         auto pdf = static_cast<RooAbsPdf *>(arg);
         coefCache[i] = pdf->expectedEvents(!refCoefNorm.empty() ? &refCoefNorm : nset);
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
         double coefSum(0);
         std::size_t i = 0;
         for (auto coefArg : coefList) {
            auto coef = static_cast<RooAbsReal *>(coefArg);
            coefCache[i] = coef->getVal(nset);
            coefSum += coefCache[i++];
         }
         if (coefSum == 0.) {
            oocoutW(&addPdf, Eval) << addPdf.ClassName() << "::updateCoefCache(" << addPdf.GetName()
                                   << ") WARNING: sum of coefficients is zero 0" << std::endl;
         } else {
            const double invCoefSum = 1. / coefSum;
            for (std::size_t j = 0; j < coefList.size(); j++) {
               coefCache[j] *= invCoefSum;
            }
         }
      } else {

         // coef[i] = coef[i] ; coef[n] = 1-SUM(coef[0...n-1])
         double lastCoef(1);
         std::size_t i = 0;
         for (auto coefArg : coefList) {
            auto coef = static_cast<RooAbsReal *>(coefArg);
            coefCache[i] = coef->getVal(nset);
            lastCoef -= coefCache[i++];
         }
         coefCache[coefList.size()] = lastCoef;

         // Treat coefficient degeneration
         const float coefDegen = lastCoef < 0. ? -lastCoef : (lastCoef > 1. ? lastCoef - 1. : 0.);
         if (coefDegen > 1.E-5) {
            coefCache[coefList.size()] = RooNaNPacker::packFloatIntoNaN(100.f * coefDegen);

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
   if ((!projectCoefs && !addPdf.normRange()) || !cache.doProjection()) {
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
