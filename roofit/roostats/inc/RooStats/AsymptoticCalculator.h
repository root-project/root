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

#include <memory>

class RooArgList;
class RooCategory;
class RooFitResult;
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

      /// use the signed (uncapped) profile likelihood test statistic.
      /// When using a one-sided test statistic, the value is not set to zero when the best fit value of the
      /// POI is beyond the tested value, but the sign of the test statistic is flipped instead
      /// (see also ProfileLikelihoodTestStat::SetSigned).
      /// It has no effect for the two-sided test statistics.
      void SetSigned(bool on = true) { fSigned = on; }

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

      /// Result of the unconditional fit to the observed data, performed by
      /// Initialize() (updated if GetHypoTest() finds a better minimum).
      /// Returns nullptr if the fit was skipped or has not been run yet.
      /// The calculator keeps ownership of the returned object.
      const RooFitResult *GetFitResultUncondObs() const { return fFitResultUncondObs.get(); }
      /// Result of the conditional fit to the observed data with the POI fixed
      /// to the tested value, from the last call to GetHypoTest().
      /// Returns nullptr if the fit was skipped or has not been run yet.
      /// The calculator keeps ownership of the returned object.
      const RooFitResult *GetFitResultCondObs() const { return fFitResultCondObs.get(); }
      /// Result of the fit to the Asimov data set with the POI fixed to the
      /// value of the alternate-model snapshot, performed by Initialize(). Since
      /// the Asimov data set is generated at that POI value, this corresponds to
      /// the unconditional minimum (updated if GetHypoTest() finds a better
      /// minimum). Returns nullptr if the fit was skipped or has not been run
      /// yet. The calculator keeps ownership of the returned object.
      const RooFitResult *GetFitResultUncondAsimov() const { return fFitResultUncondAsimov.get(); }
      /// Result of the conditional fit to the Asimov data set with the POI
      /// fixed to the tested value, from the last call to GetHypoTest().
      /// Returns nullptr if the fit was skipped or has not been run yet.
      /// The calculator keeps ownership of the returned object.
      const RooFitResult *GetFitResultCondAsimov() const { return fFitResultCondAsimov.get(); }

      static void SetPrintLevel(int level);

   private:
      bool fOneSided;                     ///< for one sided PL test statistic (upper limits)
      mutable bool fOneSidedDiscovery;    ///< for one sided PL test statistic (for discovery)
      bool fSigned = false;               ///< use signed (uncapped) PL test statistic
      bool fNominalAsimov;                ///< make Asimov at nominal parameter values
      mutable bool fIsInitialized;        ///<! flag to check if calculator is initialized
      mutable int fUseQTilde;             ///< flag to indicate if using qtilde or not (-1 (default based on RooRealVar)), 0 false, 1 (true)
      mutable double fNLLObs;
      mutable double fNLLAsimov;

      mutable RooAbsData * fAsimovData;   ///< asimov data set
      mutable RooArgSet  fAsimovGlobObs;  ///< snapshot of Asimov global observables
      mutable RooArgSet  fBestFitPoi;     ///< snapshot of best fitted POI values
      mutable RooArgSet  fBestFitParams;  ///< snapshot of all best fitted Parameter values

      mutable std::unique_ptr<RooFitResult> fFitResultUncondObs;    ///<! result of unconditional fit to observed data
      mutable std::unique_ptr<RooFitResult> fFitResultCondObs;      ///<! result of conditional fit to observed data
      mutable std::unique_ptr<RooFitResult> fFitResultUncondAsimov; ///<! result of fit to Asimov data at the alt POI
      mutable std::unique_ptr<RooFitResult> fFitResultCondAsimov;   ///<! result of conditional fit to Asimov data

      ClassDefOverride(AsymptoticCalculator,0)
   };
}

#endif
