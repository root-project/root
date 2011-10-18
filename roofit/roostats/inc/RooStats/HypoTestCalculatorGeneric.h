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

#ifndef ROOSTATS_HypoTestCalculator
#include "RooStats/HypoTestCalculator.h"
#endif

#ifndef ROOSTATS_ModelConfig
#include "RooStats/ModelConfig.h"
#endif

#ifndef ROOSTATS_TestStatistic
#include "RooStats/TestStatistic.h"
#endif

#ifndef ROOSTATS_TestStatSampler
#include "RooStats/TestStatSampler.h"
#endif

#ifndef ROOSTATS_SamplingDistribution
#include "RooStats/SamplingDistribution.h"
#endif

#ifndef ROOSTATS_HypoTestResult
#include "RooStats/HypoTestResult.h"
#endif

namespace RooStats {

   class HypoTestCalculatorGeneric : public HypoTestCalculator {

   public:
      HypoTestCalculatorGeneric(
                        const RooAbsData &data,
                        const ModelConfig &altModel,
                        const ModelConfig &nullModel,
                        TestStatSampler* sampler=0
      );


      ~HypoTestCalculatorGeneric();


   public:

      /// inherited methods from HypoTestCalculator interface
      virtual HypoTestResult* GetHypoTest() const;

      // set the model for the null hypothesis (only B)
      virtual void SetNullModel(const ModelConfig &nullModel) { fNullModel = &nullModel; }
      const RooAbsData * GetData(void) const { return fData; }
      const ModelConfig* GetNullModel(void) const { return fNullModel; }
      // set the model for the alternate hypothesis  (S+B)
      virtual void SetAlternateModel(const ModelConfig &altModel) { fAltModel = &altModel; }
      const ModelConfig* GetAlternateModel(void) const { return fAltModel; }
      // Set the DataSet
      virtual void SetData(RooAbsData &data) { fData = &data; }

      // Returns instance of TestStatSampler. Use to change properties of
      // TestStatSampler, e.g. GetTestStatSampler.SetTestSize(Double_t size);
      TestStatSampler* GetTestStatSampler(void) const { return fTestStatSampler; }

   protected:
      // should return zero (to be used later for conditional flow)
      virtual int CheckHook(void) const { return 0; }
      virtual int PreNullHook(RooArgSet* /*parameterPoint*/, double /*obsTestStat*/) const { return 0; }
      virtual int PreAltHook(RooArgSet* /*parameterPoint*/, double /*obsTestStat*/) const { return 0; }

   protected:
      const ModelConfig *fAltModel;
      const ModelConfig *fNullModel;
      const RooAbsData *fData;
      TestStatSampler *fTestStatSampler;
      TestStatSampler *fDefaultSampler;
      TestStatistic *fDefaultTestStat;

   private:
      void SetupSampler(const ModelConfig& model) const;
      void SetAdaptiveLimits(Double_t obsTestStat, Bool_t forNull) const;
      SamplingDistribution* GenerateSamplingDistribution(
         ModelConfig *thisModel,
         double obsTestStat,
         RooAbsPdf *impDens=NULL,
         const RooArgSet *impSnapshot=NULL
      ) const;


   protected:
   ClassDef(HypoTestCalculatorGeneric,1)
};
}

#endif
