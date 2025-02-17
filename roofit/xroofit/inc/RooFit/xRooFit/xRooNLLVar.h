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
class RooConstraintSum;
class RooRealVar;
class RooCmdArg;

class TGraph;
class TGraphErrors;
class TMultiGraph;
class TFile;

namespace RooStats {
class HypoTestResult;
class HypoTestInverterResult;
} // namespace RooStats

BEGIN_XROOFIT_NAMESPACE

class xRooNode;

class xRooNLLVar : public std::shared_ptr<RooAbsReal> {

public:
   struct xValueWithError : public std::pair<double, double> {
      xValueWithError(const std::pair<double, double> &in = {0, 0}) : std::pair<double, double>(in) {}
      double value() const { return std::pair<double, double>::first; }
      double error() const { return std::pair<double, double>::second; }
   };

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
      xRooFitResult(const RooFitResult &fr);
      xRooFitResult(const std::shared_ptr<xRooNode> &in,
                    const std::shared_ptr<xRooNLLVar> &nll = nullptr); // : fNode(in) { }
      const RooFitResult *operator->() const;
      //        operator std::shared_ptr<const RooFitResult>() const;
      operator const RooFitResult *() const;
      void Draw(Option_t *opt = "");

      std::shared_ptr<xRooNLLVar> nll() const { return fNll; }

      RooArgList poi()
      {
         return get()
                   ? RooArgList(*std::unique_ptr<RooAbsCollection>(get()->floatParsFinal().selectByAttrib("poi", true)))
                   : RooArgList();
      }

      // generate a conditional fit using the given poi set to the given values
      // alias is used to store the fit result in the map under a different name
      xRooFitResult cfit(const char *poiValues, const char *alias = nullptr);
      // generate the conditional fit required for an impact calculation
      xRooFitResult ifit(const char *np, bool up, bool prefit = false);
      // calculate the impact on poi due to np. if approx is true, will use the covariance approximation instead
      double impact(const char *poi, const char *np, bool up = true, bool prefit = false, bool approx = false);
      double impact(const char *np, bool up = true, bool prefit = false, bool approx = false)
      {
         auto _poi = poi();
         if (_poi.size() != 1)
            throw std::runtime_error("xRooFitResult::impact: not one POI");
         return impact(poi().contentsString().c_str(), np, up, prefit, approx);
      }

      // calculate error on poi conditional on the given NPs being held constant at their post-fit values
      // The conditional error is often presented as the difference in quadrature to the total error i.e.
      // error contribution due to conditional NPs = sqrt( pow(totError,2) - pow(condError,2) )
      double conditionalError(const char *poi, const char *nps, bool up = true, bool approx = false);

      // rank all the np based on impact ... will use the covariance approximation if full impact not available
      // the approxThreshold sets the level below which the approximation will be returned
      // e.g. set it to 0 to not do approximation
      RooArgList ranknp(const char *poi, bool up = true, bool prefit = false,
                        double approxThreshold = std::numeric_limits<double>::infinity());
      // version that assumes only one parameter is poi
      RooArgList
      ranknp(bool up = true, bool prefit = false, double approxThreshold = std::numeric_limits<double>::infinity())
      {
         auto _poi = poi();
         if (_poi.size() != 1)
            throw std::runtime_error("xRooFitResult::ranknp: not one POI");
         return ranknp(poi().contentsString().c_str(), up, prefit, approxThreshold);
      }

      std::shared_ptr<xRooNode> fNode;
      std::shared_ptr<xRooNLLVar> fNll;

      std::shared_ptr<std::map<std::string, xRooFitResult>> fCfits;
   };

   xRooFitResult minimize(const std::shared_ptr<ROOT::Fit::FitConfig> & = nullptr);

   void SetFitConfig(const std::shared_ptr<ROOT::Fit::FitConfig> &in) { fFitConfig = in; }
   std::shared_ptr<ROOT::Fit::FitConfig> fitConfig(); // returns fit config, or creates a default one if not existing
   ROOT::Math::IOptions *fitConfigOptions(); // return pointer to non-const version of the options inside the fit config

   class xRooHypoPoint : public TNamed {
   public:
      xRooHypoPoint(std::shared_ptr<RooStats::HypoTestResult> htr = nullptr, const RooAbsCollection *_coords = nullptr);
      static std::set<int> allowedStatusCodes;
      void Print(Option_t *opt = "") const override;
      void Draw(Option_t *opt = "") override;

      // status bitmask of the available fit results
      // 0 = all ok
      int status() const;

      xValueWithError pll(bool readOnly = false);      // observed test statistic value
      xValueWithError sigma_mu(bool readOnly = false); // estimate of sigma_mu parameter
      std::shared_ptr<const RooFitResult> ufit(bool readOnly = false);
      std::shared_ptr<const RooFitResult> cfit_null(bool readOnly = false);
      std::shared_ptr<const RooFitResult> cfit_alt(bool readOnly = false);
      std::shared_ptr<const RooFitResult> cfit_lbound(bool readOnly = false); // cfit @ the lower bound of mu
      std::shared_ptr<const RooFitResult> gfit() { return fGenFit; }          // non-zero if data was generated

      std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> fData;
      std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> data();

      xValueWithError getVal(const char *what);

      // leave nSigma=NaN for observed p-value
      xValueWithError pNull_asymp(double nSigma = std::numeric_limits<double>::quiet_NaN());
      xValueWithError pAlt_asymp(double nSigma = std::numeric_limits<double>::quiet_NaN());
      xValueWithError pCLs_asymp(double nSigma = std::numeric_limits<double>::quiet_NaN());
      xValueWithError ts_asymp(double nSigma = std::numeric_limits<double>::quiet_NaN()); // test statistic value

      xValueWithError pNull_toys(double nSigma = std::numeric_limits<double>::quiet_NaN());
      xValueWithError pAlt_toys(double nSigma = std::numeric_limits<double>::quiet_NaN());
      xValueWithError pCLs_toys(double nSigma = std::numeric_limits<double>::quiet_NaN())
      {
         if (fNullVal() == fAltVal())
            return std::pair<double, double>(1, 0); // by construction
         auto null = pNull_toys(nSigma);
         auto alt = pAlt_toys(nSigma);
         double nom = (null.first == 0) ? 0 : null.first / alt.first;
         // double up = (null.first + null.second == 0) ? 0 : ((alt.first-alt.second<=0) ?
         // std::numeric_limits<double>::infinity() : (null.first + null.second)/(alt.first - alt.second)); double down
         // = (null.first - null.second == 0) ? 0 : (null.first - null.second)/(alt.first + alt.second);
         //  old way ... now doing like in pCLs_asymp by calculating the two variations ... but this is pessimistic
         //  assumes p-values are anticorrelated!
         //  so reverting to old
         return std::pair<double, double>(nom, (alt.first - alt.second <= 0)
                                                  ? std::numeric_limits<double>::infinity()
                                                  : (sqrt(pow(null.second, 2) + pow(alt.second * nom, 2)) / alt.first));
         // return std::pair(nom,std::max(std::abs(up - nom), std::abs(down - nom)));
      }
      xValueWithError ts_toys(double nSigma = std::numeric_limits<double>::quiet_NaN()); // test statistic value

      // Create a HypoTestResult representing the current state of this hypoPoint
      RooStats::HypoTestResult result();

      xRooHypoPoint generateNull(int seed = 0);
      xRooHypoPoint generateAlt(int seed = 0);

      void
      addNullToys(int nToys = 1, int seed = 0, double target = std::numeric_limits<double>::quiet_NaN(),
                  double target_nSigma = std::numeric_limits<double>::quiet_NaN()); // if seed=0 will use a random seed
      void
      addAltToys(int nToys = 1, int seed = 0, double target = std::numeric_limits<double>::quiet_NaN(),
                 double target_nSigma = std::numeric_limits<double>::quiet_NaN()); // if seed=0 will use a random seed
      void
      addCLsToys(int nToys = 1, int seed = 0, double target = std::numeric_limits<double>::quiet_NaN(),
                 double target_nSigma = std::numeric_limits<double>::quiet_NaN()); // if seed=0 will use a random seed

      RooArgList poi() const;
      RooArgList alt_poi() const; // values of the poi in the alt hypothesis (will be nans if not defined)
      RooRealVar &mu_hat();       // throws exception if ufit not available

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

      std::shared_ptr<const RooFitResult> fUfit, fNull_cfit, fAlt_cfit, fLbound_cfit;
      std::shared_ptr<const RooFitResult> fGenFit; // if the data was generated, this is the fit is was generated from
      bool isExpected = false;                     // if genFit, flag says is asimov or not

      std::shared_ptr<xRooHypoPoint>
         fAsimov; // same as this point but pllType is twosided and data is expected post alt-fit

      // first is seed, second is ts value, third is weight
      std::vector<std::tuple<int, double, double>> nullToys; // would have to save these vectors for specific: null_cfit
                                                             // (genPoint), ufit, poiName, pllType, nullVal
      std::vector<std::tuple<int, double, double>> altToys;

      std::shared_ptr<xRooNLLVar> nllVar = nullptr; // hypopoints get a copy
      std::shared_ptr<RooStats::HypoTestResult> hypoTestResult = nullptr;
      std::shared_ptr<const RooFitResult> retrieveFit(int type);

      TString tsTitle(bool inWords = false) const;

   private:
      xValueWithError pX_toys(bool alt, double nSigma = std::numeric_limits<double>::quiet_NaN());
      size_t addToys(bool alt, int nToys, int initialSeed = 0, double target = std::numeric_limits<double>::quiet_NaN(),
                     double target_nSigma = std::numeric_limits<double>::quiet_NaN(), bool targetCLs = false,
                     double relErrThreshold = 2., size_t maxToys = 10000);
   };

   // use alt_value = nan to skip the asimov calculations
   xRooHypoPoint hypoPoint(const char *parName, double value,
                           double alt_value = std::numeric_limits<double>::quiet_NaN(),
                           const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown);
   // same as above but specify parNames and values in a string
   xRooHypoPoint hypoPoint(const char *parValues, double alt_value, const xRooFit::Asymptotics::PLLType &pllType);
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
      xRooHypoSpace(const RooStats::HypoTestInverterResult *result);

      bool AddModel(const xRooNode &pdf, const char *validity = "");

      void LoadFits(const char *apath);

      // A points over given parameter, number of points between low and high
      int AddPoints(const char *parName, size_t nPoints, double low, double high);

      void Print(Option_t *opt = "") const override;

      void Draw(Option_t *opt = "") override;

      RooArgList poi();
      std::shared_ptr<RooArgSet> pars() const { return fPars; };
      RooArgList axes() const;

      xRooHypoPoint &AddPoint(double value);            // adds by using the first axis var
      xRooHypoPoint &AddPoint(const char *coords = ""); // adds a new point at given coords or returns existing

      xRooHypoPoint &point(size_t i) { return at(i); }

      // build a TGraphErrors of pValues over the existing points
      // opt should include any of the following:
      //  cls: do pCLs, otherwise do pNull
      //  expX: do expected, X sigma (use +X or -X for contour, otherwise will return band unless X=0)
      //  toys: pvalues from available toys
      //  readonly: don't compute anything, just return available values
      std::shared_ptr<TGraphErrors> graph(const char *opt) const;

      // return a TMultiGraph containing the set of graphs for a particular visualization
      std::shared_ptr<TMultiGraph> graphs(const char *opt);

      // will evaluate more points until limit is below given relative uncert
      xValueWithError findlimit(const char *opt, double relUncert = std::numeric_limits<double>::infinity(),
                                unsigned int maxTries = 20);

      // get currently available limit, with error. Use nSigma = nan for observed limit
      xValueWithError limit(const char *type = "cls", double nSigma = std::numeric_limits<double>::quiet_NaN()) const;
      int scan(const char *type, size_t nPoints, double low = std::numeric_limits<double>::quiet_NaN(),
               double high = std::numeric_limits<double>::quiet_NaN(),
               const std::vector<double> &nSigmas = {0, 1, 2, -1, -2, std::numeric_limits<double>::quiet_NaN()},
               double relUncert = 0.1);
      int scan(const char *type = "cls",
               const std::vector<double> &nSigmas = {0, 1, 2, -1, -2, std::numeric_limits<double>::quiet_NaN()},
               double relUncert = 0.1)
      {
         return scan(type, 0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                     nSigmas, relUncert);
      }
      int scan(const char *type, double nSigma, double relUncert = 0.1)
      {
         return scan(type, std::vector<double>{nSigma}, relUncert);
      }

      // key is nSigma or "obs" for observed
      // will only do obs if "obs" dataset is not a generated dataset
      std::map<std::string, xValueWithError>
      limits(const char *opt = "cls",
             const std::vector<double> &nSigmas = {0, 1, 2, -1, -2, std::numeric_limits<double>::quiet_NaN()},
             double relUncert = std::numeric_limits<double>::infinity());

      std::shared_ptr<xRooNode> pdf(const RooAbsCollection &parValues) const;
      std::shared_ptr<xRooNode> pdf(const char *parValues = "") const;

      // caller needs to take ownership of the returned object
      RooStats::HypoTestInverterResult *result();

   private:
      // estimates where corresponding pValues graph becomes equal to 0.05
      // linearly interpolates log(pVal) when obtaining limits.
      // returns value and error
      static xValueWithError GetLimit(const TGraph &pValues, double target = std::numeric_limits<double>::quiet_NaN());

      static RooArgList toArgs(const char *str);

      xRooFit::Asymptotics::PLLType fTestStatType = xRooFit::Asymptotics::Unknown;
      std::shared_ptr<RooArgSet> fPars;

      std::map<std::shared_ptr<xRooNode>, std::shared_ptr<xRooNLLVar>> fNlls; // existing NLL functions of added pdfs;

      std::set<std::pair<std::shared_ptr<RooArgList>, std::shared_ptr<xRooNode>>> fPdfs;

      std::shared_ptr<TFile> fFitDb;
   };

   xRooHypoSpace hypoSpace(const char *parName, int nPoints, double low, double high,
                           double alt_value = std::numeric_limits<double>::quiet_NaN(),
                           const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown);
   xRooHypoSpace hypoSpace(const char *parName = "",
                           const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown,
                           double alt_value = std::numeric_limits<double>::quiet_NaN());
   xRooHypoSpace hypoSpace(int nPoints, double low, double high,
                           double alt_value = std::numeric_limits<double>::quiet_NaN(),
                           const xRooFit::Asymptotics::PLLType &pllType = xRooFit::Asymptotics::Unknown);
   xRooHypoSpace hypoSpace(const char *parName, xRooFit::TestStatistic::Type tsType, int nPoints = 0)
   {
      return hypoSpace(parName, int(tsType), nPoints, -std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity());
   }

   std::shared_ptr<RooArgSet> pars(bool stripGlobalObs = true) const;

   void Draw(Option_t *opt = "");

   TObject *Scan(const RooArgList &scanPars, const std::vector<std::vector<double>> &coords,
                 const RooArgList &profilePars = RooArgList());
   TObject *Scan(const char *scanPars, const std::vector<std::vector<double>> &coords,
                 const RooArgList &profilePars = RooArgList());
   TObject *Scan(const char *scanPars, size_t nPoints, double low, double high, size_t nPointsY, double ylow,
                 double yhigh, const RooArgList &profilePars = RooArgList())
   {
      std::vector<std::vector<double>> coords;
      if (nPoints) {
         double step = (high - low) / (nPoints);
         for (size_t i = 0; i < nPoints; i++) {
            std::vector<double> coord({low + step * i});
            if (nPointsY) {
               double stepy = (yhigh - ylow) / (nPointsY);
               for (size_t j = 0; j < nPointsY; j++) {
                  coord.push_back({ylow + stepy * j});
                  coords.push_back(coord);
                  coord.resize(1);
               }
            } else {
               coords.push_back(coord);
            }
         }
      }
      return Scan(scanPars, coords, profilePars);
   }
   TObject *
   Scan(const char *scanPars, size_t nPoints, double low, double high, const RooArgList &profilePars = RooArgList())
   {
      return Scan(scanPars, nPoints, low, high, 0, 0, 0, profilePars);
   }

   std::shared_ptr<RooAbsReal> func() const; // will assign globs when called
   std::shared_ptr<RooAbsPdf> pdf() const { return fPdf; }
   RooAbsData *data() const; // returns the data hidden inside the NLLVar if there is some
   const RooAbsCollection *globs() const { return fGlobs.get(); }

   // NLL = mainTerm + constraintTerm
   // mainTerm = sum( entryVals ) + extendedTerm + simTerm [+ binnedDataTerm if activated binnedL option]
   // this is what it should be, at least

   // total nll should be all these values + constraint term + extended term + simTerm [+binnedDataTerm if activated
   // binnedL option]
   /*RooAbsReal *mainTerm() const;*/
   RooConstraintSum *constraintTerm() const;

   double mainTermVal() const;
   double constraintTermVal() const;

   double getEntryVal(size_t entry) const; // get the Nll value for a specific entry
   double extendedTermVal() const;
   double simTermVal() const;
   double binnedDataTermVal() const;
   double getEntryBinWidth(size_t entry) const;

   double ndof() const;
   double saturatedVal() const;
   [[deprecated("Use saturatedConstraintTermVal()")]] double saturatedConstraintTerm() const
   {
      return saturatedConstraintTermVal();
   }
   double saturatedConstraintTermVal() const;
   [[deprecated("Use saturatedMainTermVal()")]] double saturatedMainTerm() const { return saturatedMainTermVal(); }
   double saturatedMainTermVal() const;
   double pgof() const; // a goodness-of-fit pvalue based on profile likelihood of a saturated model
   double mainTermPgof() const;
   double mainTermNdof() const;

   std::set<std::string> binnedChannels() const;

   // change the dataset - will check globs are the same
   bool setData(const std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> &_data);
   bool setData(const std::shared_ptr<RooAbsData> &data, const std::shared_ptr<const RooAbsCollection> &globs)
   {
      return setData(std::make_pair(data, globs));
   }
   bool setData(const xRooNode &data);

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

namespace cling {
std::string printValue(const xRooNLLVar::xValueWithError *val);
std::string printValue(const std::map<std::string, xRooNLLVar::xValueWithError> *m);
} // namespace cling

END_XROOFIT_NAMESPACE

#endif // include guard
