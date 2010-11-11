// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Sven Kreiss   23/05/10
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HybridCalculator
#define ROOSTATS_HybridCalculator


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

   class HybridCalculator: public HypoTestCalculator {

   public:
      HybridCalculator(
			RooAbsData &data,
			ModelConfig &altModel,
			ModelConfig &nullModel,
			TestStatSampler* sampler=0
      );


      ~HybridCalculator();


   public:

      /// inherited methods from HypoTestCalculator interface
      virtual HypoTestResult* GetHypoTest() const;

      // set the model for the null hypothesis (only B)
      virtual void SetNullModel(const ModelConfig &nullModel) { fNullModel = nullModel; }
      // set the model for the alternate hypothesis  (S+B)
      virtual void SetAlternateModel(const ModelConfig &altModel) { fAltModel = altModel; }
      // Set the DataSet
      virtual void SetData(RooAbsData &data) { fData = data; }

      // Override the distribution used for marginalizing nuisance parameters that is infered from ModelConfig
      virtual void ForcePriorNuisanceNull(RooAbsPdf& priorNuisance) { fPriorNuisanceNull = &priorNuisance; }
      virtual void ForcePriorNuisanceAlt(RooAbsPdf& priorNuisance) { fPriorNuisanceAlt = &priorNuisance; }

      // Returns instance of TestStatSampler. Use to change properties of
      // TestStatSampler, e.g. GetTestStatSampler.SetTestSize(Double_t size);
      TestStatSampler* GetTestStatSampler(void) { return fTestStatSampler; }

      // Enable adaptive sampling (for use with toymcsampler).
      // A value different from 0.0 enables adaptive sampling.
      void SetAdaptiveSampling(Double_t toysInTails = 0.0) { fToysInTails = toysInTails; }

      // Wrapper for ToyMCSampler function when adaptive sampling is used.
      // See doc in toymcsampler.
      void SetMaxToys(Double_t t);

      // sets importance density and snapshot (optional)
      void SetNullImportanceDensity(RooAbsPdf *p, RooArgSet *s = NULL) {
         fNullImportanceDensity = p;
         fNullImportanceSnapshot = s;
      }

   private:
      void SetupSampler(ModelConfig& model) const;
      void SetAdaptiveLimits(Double_t obsTestStat, Bool_t forNull) const;
      SamplingDistribution* GenerateSamplingDistribution(
         ModelConfig *thisModel,
         double obsTestStat,
         RooAbsPdf *impDens=NULL,
         RooArgSet *impSnapshot=NULL
      ) const;

      ModelConfig &fAltModel;
      ModelConfig &fNullModel;
      RooAbsData &fData;
      RooAbsPdf *fPriorNuisanceNull;
      RooAbsPdf *fPriorNuisanceAlt;
      TestStatSampler* fTestStatSampler;
      TestStatSampler* fDefaultSampler;
      TestStatistic* fDefaultTestStat;

      Double_t fToysInTails; // a value different from 0.0 enables adaptive sampling

      RooAbsPdf *fNullImportanceDensity;
      RooArgSet *fNullImportanceSnapshot;

   protected:
   ClassDef(HybridCalculator,1)
};
}

#endif
