// @(#)root/roostats:$Id$
// Author: Sven Kreiss, Kyle Cranmer   Nov 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_AsymptoticCalculator
#define ROOSTATS_AsymptoticCalculator

#include "RooStats/HypoTestCalculatorGeneric.h"
#include "RooArgSet.h"
#include "Rtypes.h"

class RooArgList;
class RooCategory;
class RooRealVar;
class RooPoisson;
class RooProdPdf;


namespace RooStats {

   class AsymptoticCalculator : public HypoTestCalculatorGeneric {

   public:
      AsymptoticCalculator(
         RooAbsData &data,  // need to pass non-const since RooAbsPdf::fitTo takes a non-const data set
         const ModelConfig &altModel,
         const ModelConfig &nullModel,
         bool nominalAsimov = false
         );

      ~AsymptoticCalculator() override {
      }

      /// initialize the calculator by performing a global fit and make the Asimov data set
      bool Initialize() const;

      /// re-implement HypoTest computation using the asymptotic
      HypoTestResult *GetHypoTest() const override;

      /// Make Asimov data.
      static RooAbsData * MakeAsimovData( RooAbsData & data, const ModelConfig & model,  const RooArgSet & poiValues, RooArgSet & globObs, const RooArgSet * genPoiValues = nullptr);


      /// Make a nominal Asimov data set from a model.
      static RooAbsData * MakeAsimovData( const ModelConfig & model,  const RooArgSet & allParamValues, RooArgSet & globObs);



      static RooAbsData * GenerateAsimovData(const RooAbsPdf & pdf, const RooArgSet & observables );

      /// function given the null and the alt p value - return the expected one given the N - sigma value
      static double GetExpectedPValues(double pnull, double palt, double nsigma, bool usecls, bool oneSided = true );

      /// set test statistic for one sided (upper limits)
      void SetOneSided(bool on) { fOneSided = on; }

      /// set the test statistics for two sided (in case of upper limits
      /// for discovery does not make really sense)
      void SetTwoSided() { fOneSided = false; fOneSidedDiscovery = false;}

      /// set the test statistics for one-sided discovery
      void SetOneSidedDiscovery(bool on) { fOneSidedDiscovery = on; }

      /// re-implementation of  setters since they are needed to re-initialize the calculator
      void SetNullModel(const ModelConfig &nullModel) override {
         HypoTestCalculatorGeneric::SetNullModel(nullModel);
         fIsInitialized = false;
      }
      void SetAlternateModel(const ModelConfig &altModel) override {
         HypoTestCalculatorGeneric::SetAlternateModel(altModel);
         fIsInitialized = false;
      }
      void SetData(RooAbsData &data) override {
         HypoTestCalculatorGeneric::SetData(data);
         fIsInitialized = false;
      }


      bool IsTwoSided() const { return (!fOneSided && !fOneSidedDiscovery); }
      bool IsOneSidedDiscovery() const { return fOneSidedDiscovery; }


      /// set using of qtilde, by default is controlled if RoORealVar is limited or not
      void SetQTilde(bool on) { fUseQTilde = on; }

      /// return snapshot of the best fit parameter
      const RooArgSet & GetBestFitPoi() const { return fBestFitPoi; }
      /// return best fit parameter (firs of poi)
      const RooRealVar * GetMuHat() const { return dynamic_cast<RooRealVar*>(fBestFitPoi.first()); }
      /// return best fit value for all parameters
      const RooArgSet & GetBestFitParams() const { return fBestFitPoi; }

      static void SetPrintLevel(int level);

   protected:
      // // configure TestStatSampler for the Null run
      // int PreNullHook(RooArgSet *parameterPoint, double obsTestStat) const;

      // // configure TestStatSampler for the Alt run
      // int PreAltHook(RooArgSet *parameterPoint, double obsTestStat) const;


      static RooAbsData * GenerateAsimovDataSinglePdf(const RooAbsPdf & pdf, const RooArgSet & obs,  const RooRealVar & weightVar,
                                                      RooCategory * channelCat = nullptr);

      static RooAbsData * GenerateCountingAsimovData(RooAbsPdf & pdf, const RooArgSet & obs,  const RooRealVar & weightVar,
                                                      RooCategory * channelCat = nullptr);


      static void FillBins(const RooAbsPdf & pdf, const RooArgList &obs, RooAbsData & data, int &index,  double
                           &binVolume, int &ibin);

      static double EvaluateNLL(RooAbsPdf & pdf, RooAbsData& data, const RooArgSet * condObs, const RooArgSet * globObs, const RooArgSet *poiSet = nullptr );

      static bool SetObsToExpected(RooAbsPdf &pdf, const RooArgSet &obs);
      static bool SetObsToExpected(RooProdPdf &prod, const RooArgSet &obs);

   protected:
      ClassDefOverride(AsymptoticCalculator,2)

   private:

      bool fOneSided;                     ///< for one sided PL test statistic (upper limits)
      mutable bool fOneSidedDiscovery;    ///< for one sided PL test statistic (for discovery)
      bool fNominalAsimov;                ///< make Asimov at nominal parameter values
      mutable bool fIsInitialized;        ///<! flag to check if calculator is initialized
      mutable int fUseQTilde;             ///< flag to indicate if using qtilde or not (-1 (default based on RooRealVar)), 0 false, 1 (true)
      static int fgPrintLevel;            ///< control print level  (0 minimal, 1 normal, 2 debug)
      mutable double fNLLObs;
      mutable double fNLLAsimov;

      mutable RooAbsData * fAsimovData;   ///< asimov data set
      mutable RooArgSet  fAsimovGlobObs;  ///< snapshot of Asimov global observables
      mutable RooArgSet  fBestFitPoi;     ///< snapshot of best fitted POI values
      mutable RooArgSet  fBestFitParams;  ///< snapshot of all best fitted Parameter values


   };
}

#endif
