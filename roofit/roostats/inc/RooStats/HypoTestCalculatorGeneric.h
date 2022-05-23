// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Sven Kreiss   23/05/10
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HypoTestCalculatorGeneric
#define ROOSTATS_HypoTestCalculatorGeneric


#ifndef ROOT_Rtypes
#include "Rtypes.h" // necessary for TNamed
#endif

#include "RooStats/HypoTestCalculator.h"

#include "RooStats/ModelConfig.h"

#include "RooStats/TestStatistic.h"

#include "RooStats/TestStatSampler.h"

#include "RooStats/SamplingDistribution.h"

#include "RooStats/HypoTestResult.h"


namespace RooStats {

   class HypoTestCalculatorGeneric : public HypoTestCalculator {

   public:
      HypoTestCalculatorGeneric(
                        const RooAbsData &data,
                        const ModelConfig &altModel,
                        const ModelConfig &nullModel,
                        TestStatSampler* sampler=0
      );


      ~HypoTestCalculatorGeneric() override;


   public:

      /// inherited methods from HypoTestCalculator interface
      HypoTestResult* GetHypoTest() const override;

      /// set the model for the null hypothesis (only B)
      void SetNullModel(const ModelConfig &nullModel) override { fNullModel = &nullModel; }
      const RooAbsData * GetData(void) const { return fData; }
      const ModelConfig* GetNullModel(void) const { return fNullModel; }
      virtual const RooArgSet* GetFitInfo() const { return nullptr; }
      /// Set the model for the alternate hypothesis  (S+B)
      void SetAlternateModel(const ModelConfig &altModel) override { fAltModel = &altModel; }
      const ModelConfig* GetAlternateModel(void) const { return fAltModel; }
      /// Set the DataSet
      void SetData(RooAbsData &data) override { fData = &data; }

      /// Returns instance of TestStatSampler. Use to change properties of
      /// TestStatSampler, e.g. GetTestStatSampler.SetTestSize(double size);
      TestStatSampler* GetTestStatSampler(void) const { return fTestStatSampler; }

      /// Set this for re-using always the same toys for alternate hypothesis in
      /// case of calls at different null parameter points
      /// This is useful to get more stable bands when running the HypoTest inversion
      void UseSameAltToys();


   protected:
      // should return zero (to be used later for conditional flow)
      virtual int CheckHook(void) const { return 0; }
      virtual int PreNullHook(RooArgSet* /*parameterPoint*/, double /*obsTestStat*/) const { return 0; }
      virtual int PreAltHook(RooArgSet* /*parameterPoint*/, double /*obsTestStat*/) const { return 0; }
      virtual void PreHook() const { }
      virtual void PostHook() const { }

   protected:
      const ModelConfig *fAltModel;
      const ModelConfig *fNullModel;
      const RooAbsData *fData;
      TestStatSampler *fTestStatSampler;
      TestStatSampler *fDefaultSampler;
      TestStatistic *fDefaultTestStat;

      unsigned int fAltToysSeed;   // to have same toys for alternate

   private:
      void SetupSampler(const ModelConfig& model) const;
      void SetAdaptiveLimits(double obsTestStat, bool forNull) const;
      SamplingDistribution* GenerateSamplingDistribution(
         ModelConfig *thisModel,
         double obsTestStat,
         RooAbsPdf *impDens=nullptr,
         const RooArgSet *impSnapshot=nullptr
      ) const;


   protected:
   ClassDefOverride(HypoTestCalculatorGeneric,2)
};
}

#endif
