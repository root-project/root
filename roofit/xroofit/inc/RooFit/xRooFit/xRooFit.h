/*
 * Project: xRooFit
 * Author:
 *   Will Buttinger, RAL 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */
#include "Config.h"

// when not using the namespace will use the once pragma.
// when using the namespace (as happens in the ROOT build of xRooFit) then
// will effectively use an include guard
#ifdef XROOFIT_USE_PRAGMA_ONCE
#pragma once
#endif
#if (!defined(XROOFIT_USE_PRAGMA_ONCE) && !defined(XROOFIT_XROOFIT_H)) || \
   (defined(XROOFIT_USE_PRAGMA_ONCE) && !defined(XROOFIT_XROOFIT_H_XROOFIT))
#ifndef XROOFIT_USE_PRAGMA_ONCE
#define XROOFIT_XROOFIT_H
#else
#define XROOFIT_XROOFIT_H_XROOFIT
// even with using pragma once, need include guard otherwise cannot include this header
// as part of an interpreted file ... the other headers in xRooFit are similarly affected
// however for now users of xRooFit should only need to include the main xRooFit header to use it all
// in future we should try removing the pragma once altogether (undef XROOFIT_USE_PRAGMA_ONCE)
// and see if it has negative consequences anywhere
#endif

/**
 * This is the main include for the xRooFit project.
 * Including this should give you access to all xRooFit features
 */

class RooAbsData;
class RooAbsCollection;
class RooFitResult;
class RooAbsPdf;
class RooAbsReal;
class RooLinkedList;
class RooWorkspace;

#include "Fit/FitConfig.h"

#include "RooCmdArg.h"
#include "TNamed.h"

class TCanvas;

#include <memory>

BEGIN_XROOFIT_NAMESPACE

class xRooNLLVar;

class xRooFit {

public:
   static const char *GetVersion();
   static const char *GetVersionDate();

   // Extra options for NLL creation:
   static RooCmdArg ReuseNLL(bool flag); // if should try to reuse the NLL object when it changes dataset
   static RooCmdArg Tolerance(double value);
   static RooCmdArg StrategySequence(const char *stratSeq); // control minimization strategy sequence
   static RooCmdArg MaxIterations(int nIterations);

   static constexpr double OBS = std::numeric_limits<double>::quiet_NaN();

   // Helper function for matching precision of a value and its error
   static std::pair<double, double> matchPrecision(const std::pair<double, double> &in);

   // Static methods that work with the 'first class' object types:
   //    Pdfs: RooAbsPdf
   //    Datasets: std::pair<RooAbsData,const RooAbsCollection>
   //    NLLOptions: RooLinkedList
   //    FitOptions: ROOT::Fit::FitConfig

   // fit result flags in its constPars list which are global observables with the "global" attribute
   static std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>>
   generateFrom(RooAbsPdf &pdf, const RooFitResult &fr, bool expected = false, int seed = 0);
   static std::shared_ptr<const RooFitResult>
   fitTo(RooAbsPdf &pdf, const std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> &data,
         const RooLinkedList &nllOpts, const ROOT::Fit::FitConfig &fitConf);
   static std::shared_ptr<const RooFitResult> fitTo(RooAbsPdf &pdf,
                                                    const std::pair<RooAbsData *, const RooAbsCollection *> &data,
                                                    const RooLinkedList &nllOpts, const ROOT::Fit::FitConfig &fitConf);

   static xRooNLLVar createNLL(const std::shared_ptr<RooAbsPdf> pdf, const std::shared_ptr<RooAbsData> data,
                               const RooLinkedList &nllOpts);
   static xRooNLLVar createNLL(RooAbsPdf &pdf, RooAbsData *data, const RooLinkedList &nllOpts);
   static xRooNLLVar createNLL(RooAbsPdf &pdf, RooAbsData *data, const RooCmdArg &arg1 = RooCmdArg::none(),
                               const RooCmdArg &arg2 = RooCmdArg::none(), const RooCmdArg &arg3 = RooCmdArg::none(),
                               const RooCmdArg &arg4 = RooCmdArg::none(), const RooCmdArg &arg5 = RooCmdArg::none(),
                               const RooCmdArg &arg6 = RooCmdArg::none(), const RooCmdArg &arg7 = RooCmdArg::none(),
                               const RooCmdArg &arg8 = RooCmdArg::none());

   static std::shared_ptr<ROOT::Fit::FitConfig> createFitConfig(); // obtain instance of default fit configuration
   static std::shared_ptr<RooLinkedList> createNLLOptions();       // obtain instance of default nll options
   static std::shared_ptr<RooLinkedList> defaultNLLOptions();      // access default NLL options for modifications
   static std::shared_ptr<ROOT::Fit::FitConfig> defaultFitConfig();
   static ROOT::Math::IOptions *defaultFitConfigOptions();

   static std::shared_ptr<const RooFitResult> minimize(RooAbsReal &nll,
                                                       const std::shared_ptr<ROOT::Fit::FitConfig> &fitConfig = nullptr,
                                                       const std::shared_ptr<RooLinkedList> &nllOpts = nullptr);
   static int minos(RooAbsReal &nll, const RooFitResult &ufit, const char *parName = "",
                    const std::shared_ptr<ROOT::Fit::FitConfig> &_fitConfig = nullptr);

   // this class is used to store a shared_ptr in a TDirectory's List, so that retrieval of cached fits
   // can share the fit result (and avoid re-reading from disk as well)
   class StoredFitResult : public TNamed {
   public:
      StoredFitResult(RooFitResult *_fr);
      StoredFitResult(const std::shared_ptr<RooFitResult> &_fr);

   public:
      std::shared_ptr<RooFitResult> fr; //!
      ClassDef(StoredFitResult, 0)
   };

   // can't use an enum class because pyROOT doesn't handle it
   class TestStatistic {
   public:
      enum Type {
         // basic likelihood ratio
         tmu = -1,
         // upper limit test statistics
         qmu = -2,
         qmutilde = -3,
         // discovery test statistics
         q0 = -4,
         uncappedq0 = -5,
         u0 = -5
      };
   };

   class Asymptotics {

   public:
      typedef std::vector<std::pair<double, int>> IncompatFunc;

      enum PLLType {
         TwoSided = 0,
         OneSidedPositive, // for exclusions
         OneSidedNegative, // for discovery
         OneSidedAbsolute, // for exclusions by magnitude
         Uncapped,         // for discovery with interest in deficits as well as excesses
         Unknown
      };

      // The incompatibility function (taking mu_hat as an input) is defined by its transitions
      // it takes values of -1, 0, or 1 ... when it 0 that means mu_hat is compatible with the hypothesis
      // Standard incompatibility functions are parameterized by mu
      // Note: the default value is taken to be 1, so an empty vector is function=1
      static IncompatFunc IncompatibilityFunction(const PLLType &type, double mu)
      {
         std::vector<std::pair<double, int>> out;
         if (type == TwoSided) {
            // standard PLL
         } else if (type == OneSidedPositive) {
            out.emplace_back(std::make_pair(mu, 0)); // becomes compatible @ mu_hat = mu
         } else if (type == OneSidedNegative) {
            out.emplace_back(std::make_pair(-std::numeric_limits<double>::infinity(), 0)); // compatible at -inf
            out.emplace_back(std::make_pair(mu, 1)); // becomes incompatible at mu_hat = mu
         } else if (type == OneSidedAbsolute) {
            out.emplace_back(std::make_pair(-std::numeric_limits<double>::infinity(), 0)); // compatible at -inf
            out.emplace_back(std::make_pair(-mu, 1)); // incompatible @ mu_hat = -mu
            out.emplace_back(std::make_pair(mu, 0));  // compatible again @ mu_hat = mu
         } else if (type == Uncapped) {
            out.emplace_back(std::make_pair(-std::numeric_limits<double>::infinity(), -1)); // reversed at -inf
            out.emplace_back(std::make_pair(mu, 1)); // becomes normal @ mu_hat = mu
         } else {
            throw std::runtime_error("Unknown PLL Type");
         }
         return out;
      }

      // inverse of PValue function
      static double k(const IncompatFunc &compatRegions, double pValue, double poiVal, double poiPrimeVal,
                      double sigma_mu = 0, double mu_low = -std::numeric_limits<double>::infinity(),
                      double mu_high = std::numeric_limits<double>::infinity());

      static double k(const PLLType &pllType, double pValue, double mu, double mu_prime, double sigma_mu = 0,
                      double mu_low = -std::numeric_limits<double>::infinity(),
                      double mu_high = std::numeric_limits<double>::infinity())
      {
         return k(IncompatibilityFunction(pllType, mu), pValue, mu, mu_prime, sigma_mu, mu_low, mu_high);
      }

      // Recommend sigma_mu = |mu - mu_prime|/sqrt(pll_mu(asimov_mu_prime))
      static double PValue(const IncompatFunc &compatRegions, double k, double mu, double mu_prime, double sigma_mu = 0,
                           double mu_low = -std::numeric_limits<double>::infinity(),
                           double mu_high = std::numeric_limits<double>::infinity());

      static double PValue(const PLLType &pllType, double k, double mu, double mu_prime, double sigma_mu = 0,
                           double mu_low = -std::numeric_limits<double>::infinity(),
                           double mu_high = std::numeric_limits<double>::infinity())
      {
         return PValue(IncompatibilityFunction(pllType, mu), k, mu, mu_prime, sigma_mu, mu_low, mu_high);
      }

      static double Phi_m(double mu, double mu_prime, double a, double sigma, const IncompatFunc &compatRegions);

      static int CompatFactor(const IncompatFunc &func, double mu_hat);

      static int CompatFactor(int type, double mu, double mu_hat)
      {
         return CompatFactor(IncompatibilityFunction((PLLType)type, mu), mu_hat);
      }

      // converts pvalues to significances and finds where they equal the target pvalue
      // return is x-axis value with potentially an error on that value if input pVals had errors
      // static RooRealVar FindLimit(TGraph *pVals, double target_pVal = 0.05);
   };

   static std::shared_ptr<RooLinkedList> sDefaultNLLOptions;
   static std::shared_ptr<ROOT::Fit::FitConfig> sDefaultFitConfig;

   // Run hypothesis test(s) on the given pdf
   // Uses hypoPoint binning on model parameters to determine points to scan
   // if hypoPoint binning has nBins==0 then will auto-scan (assumes CL=95%, can override with setStringAttribute)
   // TODO: specifying number of null and alt toys per point
   static TCanvas *
   hypoTest(RooWorkspace &w, const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown)
   {
      return hypoTest(w, 0, 0, pllType);
   }
   static TCanvas *hypoTest(RooWorkspace &w, int nToysNull, int nToysAlt,
                            const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown);
};

END_XROOFIT_NAMESPACE

#include "xRooHypoSpace.h"
#include "xRooNLLVar.h"
#include "xRooNode.h"

#endif // include guard
