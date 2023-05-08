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

#ifdef XROOFIT_USE_PRAGMA_ONCE
#pragma once
#endif
#if !defined(XROOFIT_XROONLLVAR_H) || defined(XROOFIT_USE_PRAGMA_ONCE)
#ifndef XROOFIT_USE_PRAGMA_ONCE
#define XROOFIT_XROONLLVAR_H
#endif

#include "xRooFit.h"

#include <RooFitResult.h>
#include <RooLinkedList.h>

#include <Fit/FitConfig.h>
#include <Math/IOptions.h>
#include <TAttFill.h>
#include <TAttLine.h>
#include <TAttMarker.h>

#include <map>
#include <set>

class RooAbsReal;
class RooAbsPdf;
class RooAbsData;
class RooAbsCollection;
class RooNLLVar;
class RooConstraintSum;
class RooRealVar;
class RooCmdArg;

class TGraph;
class TGraphErrors;

namespace RooStats {
class HypoTestResult;
class HypoTestInverterResult;
} // namespace RooStats

BEGIN_XROOFIT_NAMESPACE

class xRooNode;

class xRooNLLVar : public std::shared_ptr<RooAbsReal> {

public:
   void Print(Option_t *opt = "");

   xRooNLLVar(RooAbsPdf &pdf, const std::pair<RooAbsData *, const RooAbsCollection *> &data,
              const RooLinkedList &nllOpts = RooLinkedList());
   xRooNLLVar(const std::shared_ptr<RooAbsPdf> &pdf, const std::shared_ptr<RooAbsData> &data,
              const RooLinkedList &opts = RooLinkedList());
   xRooNLLVar(const std::shared_ptr<RooAbsPdf> &pdf,
              const std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> &data,
              const RooLinkedList &opts = RooLinkedList());

   ~xRooNLLVar();

   // whenever implicitly converted to a RooAbsReal we will make sure our globs are set
   RooAbsReal *get() const { return func().get(); }
   RooAbsReal *operator->() const { return get(); }

   void reinitialize();

   void AddOption(const RooCmdArg &opt);

   std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>>
   getData() const; // returns pointer to data and snapshot of globs
   std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>>
   generate(bool expected = false, int seed = 0);
   // std::shared_ptr<const RooFitResult> snapshot();

   class xRooFitResult : public std::shared_ptr<const RooFitResult> {
   public:
      xRooFitResult(const std::shared_ptr<xRooNode> &in); // : fNode(in) { }
      const RooFitResult *operator->() const;
      //        operator std::shared_ptr<const RooFitResult>() const;
      operator const RooFitResult *() const;
      void Draw(Option_t *opt = "");
      std::shared_ptr<xRooNode> fNode;
   };

   xRooFitResult minimize(const std::shared_ptr<ROOT::Fit::FitConfig> & = nullptr);

   void SetFitConfig(const std::shared_ptr<ROOT::Fit::FitConfig> &in) { fFitConfig = in; }
   std::shared_ptr<ROOT::Fit::FitConfig> fitConfig(); // returns fit config, or creates a default one if not existing
   ROOT::Math::IOptions *fitConfigOptions(); // return pointer to non-const version of the options inside the fit config

   class xRooHypoPoint {
   public:
      static std::set<int> allowedStatusCodes;
      void Print();
      void Draw(Option_t *opt = "");

      // status bitmask of the available fit results
      // 0 = all ok
      int status() const;

      std::pair<double, double> pll(bool readOnly = false);      // observed test statistic value
      std::pair<double, double> sigma_mu(bool readOnly = false); // estimate of sigma_mu parameter
      std::shared_ptr<const RooFitResult> ufit(bool readOnly = false);
      std::shared_ptr<const RooFitResult> cfit_null(bool readOnly = false);
      std::shared_ptr<const RooFitResult> cfit_alt(bool readOnly = false);

      std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> data;

      std::pair<double, double> getVal(const char *what);

      // leave nSigma=NaN for observed p-value
      std::pair<double, double> pNull_asymp(double nSigma = std::numeric_limits<double>::quiet_NaN());
      std::pair<double, double> pAlt_asymp(double nSigma = std::numeric_limits<double>::quiet_NaN());
      std::pair<double, double> pCLs_asymp(double nSigma = std::numeric_limits<double>::quiet_NaN());
      std::pair<double, double>
      ts_asymp(double nSigma = std::numeric_limits<double>::quiet_NaN()); // test statistic value

      std::pair<double, double> pNull_toys(double nSigma = std::numeric_limits<double>::quiet_NaN());
      std::pair<double, double> pAlt_toys(double nSigma = std::numeric_limits<double>::quiet_NaN());
      std::pair<double, double> pCLs_toys(double nSigma = std::numeric_limits<double>::quiet_NaN())
      {
         if (fNullVal() == fAltVal())
            return std::pair(1, 0); // by construction
         auto null = pNull_toys(nSigma);
         auto alt = pAlt_toys(nSigma);
         double pval = (null.first == 0) ? 0 : null.first / alt.first;
         // TODO: should do error calculation like for asymp (calulate up and down separately and then take err)
         return std::make_pair(pval, pval * sqrt(pow(null.second / null.first, 2) + pow(alt.second / alt.first, 2)));
      }
      std::pair<double, double>
      ts_toys(double nSigma = std::numeric_limits<double>::quiet_NaN()); // test statistic value

      // Create a HypoTestResult representing the current state of this hypoPoint
      RooStats::HypoTestResult result();

      xRooHypoPoint generateNull(int seed = 0);
      xRooHypoPoint generateAlt(int seed = 0);

      void addNullToys(int nToys = 1, int seed = 0); // if seed=0 will use a random seed
      void addAltToys(int nToys = 1, int seed = 0);  // if seed=0 will use a random seed

      RooArgList poi();
      RooArgList alt_poi(); // values of the poi in the alt hypothesis (will be nans if not defined)
      RooRealVar &mu_hat(); // throws exception if ufit not available

      std::shared_ptr<xRooHypoPoint>
      asimov(bool readOnly =
                false); // a two-sided hypoPoint with the alt hypothesis asimov dataset (used in sigma_mu() calculation)

      // std::string fPOIName;
      const char *fPOIName();
      xRooFit::Asymptotics::PLLType fPllType = xRooFit::Asymptotics::Unknown;
      // double fNullVal=1; double fAltVal=0;
      double fNullVal();
      double fAltVal();

      std::shared_ptr<const RooAbsCollection> coords; // pars of the nll that will be held const alongside POI

      std::shared_ptr<const RooFitResult> fUfit, fNull_cfit, fAlt_cfit;
      std::shared_ptr<const RooFitResult> fGenFit; // if the data was generated, this is the fit is was generated from
      bool isExpected = false;                     // if genFit, flag says is asimov or not

      std::shared_ptr<xRooHypoPoint>
         fAsimov; // same as this point but pllType is twosided and data is expected post alt-fit

      // first is seed, second is ts value, third is weight
      std::vector<std::tuple<int, double, double>> nullToys; // would have to save these vectors for specific: null_cfit
                                                             // (genPoint), ufit, poiName, pllType, nullVal
      std::vector<std::tuple<int, double, double>> altToys;

      std::shared_ptr<xRooNLLVar> nllVar = nullptr; // hypopoints get a copy

   private:
      std::pair<double, double> pX_toys(bool alt, double nSigma = std::numeric_limits<double>::quiet_NaN());
      void addToys(bool alt, int nToys, int initialSeed = 0);

      TString tsTitle();
   };

   // use alt_value = nan to skip the asimov calculations
   xRooHypoPoint hypoPoint(const char *parName, double value,
                           double alt_value = std::numeric_limits<double>::quiet_NaN(),
                           const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown);
   // this next method requires poi to be flagged in the model already (with "poi" attribute) .. must be exactly one
   xRooHypoPoint hypoPoint(double value, double alt_value = std::numeric_limits<double>::quiet_NaN(),
                           const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown);

   class xRooHypoSpace : public TNamed,
                         public TAttFill,
                         public TAttMarker,
                         public TAttLine,
                         public std::vector<xRooHypoPoint> {
   public:
      friend class xRooNLLVar;
      xRooHypoSpace(const char *name = "", const char *title = "");

      bool AddWorkspace(const char *wsFilename, const char *extraPars = "");

      bool AddModel(const xRooNode &pdf, const char *validity = "");

      void LoadFits(const char *apath);

      // A points over given parameter, number of points between low and high
      int AddPoints(const char *parName, size_t nPoints, double low, double high);

      void Print(Option_t *opt = "") const override;

      void Draw(Option_t *opt = "") override;

      RooArgList poi();
      std::shared_ptr<RooArgSet> pars() const { return fPars; };
      RooArgList axes() const;

      xRooHypoPoint &AddPoint(const char *coords = ""); // adds a new point at given coords or returns existing

      xRooHypoPoint &point(size_t i) { return at(i); }

      // build a TGraphErrors of pValues over the existing points
      // opt should include any of the following:
      //  cls: do pCLs, otherwise do pNull
      //  expX: do expected, X sigma (use +X or -X for contour, otherwise will return band unless X=0)
      //  toys: pvalues from available toys
      //  readonly: don't compute anything, just return available values
      std::shared_ptr<TGraphErrors> BuildGraph(const char *opt);

      // estimates where corresponding pValues graph becomes equal to 0.05
      // linearly interpolates log(pVal) when obtaining limits.
      // returns value and error
      static std::pair<double, double> GetLimit(const TGraph &pValues, double target = 0.05);

      // will evaluate more points until limit is below given relative uncert

      std::pair<double, double> FindLimit(const char *opt, double relUncert = std::numeric_limits<double>::infinity());

      // key is nSigma or "obs" for observed
      std::map<std::string, std::pair<double, double>> limits(const char *opt = "cls", double relUncert = 0.1);

      std::shared_ptr<xRooNode> pdf(const RooAbsCollection &parValues) const;
      std::shared_ptr<xRooNode> pdf(const char *parValues = "") const;

      // caller needs to take ownership of the returned object
      RooStats::HypoTestInverterResult *result();

   private:
      static RooArgList toArgs(const char *str);

      xRooFit::Asymptotics::PLLType fTestStatType = xRooFit::Asymptotics::Unknown;
      std::shared_ptr<RooArgSet> fPars;

      std::map<std::shared_ptr<xRooNode>, std::shared_ptr<xRooNLLVar>> fNlls; // existing NLL functions of added pdfs;

      std::set<std::shared_ptr<xRooNode>> fWorkspaces; // added workspaces (kept open)

      std::set<std::pair<std::shared_ptr<RooArgList>, std::shared_ptr<xRooNode>>> fPdfs;
   };

   xRooHypoSpace hypoSpace(const char *parName, int nPoints, double low, double high,
                           double alt_value = std::numeric_limits<double>::quiet_NaN(),
                           const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown);
   xRooHypoSpace
   hypoSpace(const char *parName = "", const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown);
   xRooHypoSpace hypoSpace(int nPoints, double low, double high,
                           double alt_value = std::numeric_limits<double>::quiet_NaN(),
                           const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown);

   std::shared_ptr<RooArgSet> pars(bool stripGlobalObs = true);

   void Draw(Option_t *opt = "");

   std::shared_ptr<RooAbsReal> func() const; // will assign globs when called
   std::shared_ptr<RooAbsPdf> pdf() const { return fPdf; }
   RooAbsData *data() const; // returns the data hidden inside the NLLVar if there is some

   // NLL = nllTerm + constraintTerm
   // nllTerm = sum( entryVals ) + extendedTerm + simTerm [+ binnedDataTerm if activated binnedL option]
   // this is what it should be, at least

   // total nll should be all these values + constraint term + extended term + simTerm [+binnedDataTerm if activated
   // binnedL option]
   RooNLLVar *nllTerm() const;
   RooConstraintSum *constraintTerm() const;

   double getEntryVal(size_t entry); // get the Nll value for a specific entry
   double extendedTerm() const;
   double simTerm() const;
   double binnedDataTerm() const;

   // change the dataset - will check globs are the same
   Bool_t setData(const std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> &_data);
   Bool_t setData(const std::shared_ptr<RooAbsData> &data, const std::shared_ptr<const RooAbsCollection> &globs)
   {
      return setData(std::make_pair(data, globs));
   }
   Bool_t setData(const xRooNode &data);

   // using shared ptrs everywhere, even for RooLinkedList which needs custom deleter to clear itself
   // but still work ok for assignment operations
   std::shared_ptr<RooAbsPdf> fPdf;
   std::shared_ptr<RooAbsData> fData;
   std::shared_ptr<const RooAbsCollection> fGlobs;

   std::shared_ptr<RooLinkedList> fOpts;
   std::shared_ptr<ROOT::Fit::FitConfig> fFitConfig;

   std::shared_ptr<RooAbsCollection> fFuncVars;
   std::shared_ptr<RooAbsCollection> fConstVars;
   std::shared_ptr<RooAbsCollection> fFuncGlobs;
   std::string fFuncCreationLog; // messaging from when function was last created -- to save from printing to screen

   bool kReuseNLL = true;
};

END_XROOFIT_NAMESPACE

#endif // include guard
