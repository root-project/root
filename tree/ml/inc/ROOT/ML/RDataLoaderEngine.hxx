// Author: Dante Niewenhuis, VU Amsterdam 07/2023
// Author: Kristupas Pranckietis, Vilnius University 05/2024
// Author: Nopphakorn Subsa-Ard, King Mongkut's University of Technology Thonburi (KMUTT) (TH) 08/2024
// Author: Vincenzo Eduardo Padulano, CERN 10/2024
// Author: Martin Føll, University of Oslo (UiO) & CERN 01/2026
// Author: Silia Taider, CERN 02/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_ML_RDATALOADERENGINE
#define ROOT_INTERNAL_ML_RDATALOADERENGINE

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "ROOT/ML/RBatchLoader.hxx"
#include "ROOT/ML/RBatchSink.hxx"
#include "ROOT/ML/RClusterLoader.hxx"
#include "ROOT/ML/RDatasetLoader.hxx"
#include "ROOT/ML/RFlat2DMatrix.hxx"
#include "ROOT/ML/RFlat2DMatrixOperators.hxx"
#include "ROOT/ML/RSampler.hxx"
#include "ROOT/RDF/InterfaceUtils.hxx"

// Empty namespace to create a hook for the Pythonization
namespace ROOT::Experimental::ML {
}

namespace ROOT::Experimental::Internal::ML {
/**
 \class ROOT::Experimental::Internal::ML::RDataLoaderEngine
\brief

In this class, the processes of loading clusters (see RClusterLoader) and creating batches from those clusters (see
RBatchLoader) are combined, allowing batches from the training and validation sets to be loaded directly from a dataset
in an RDataFrame.
*/

class RDataLoaderEngine {
private:
   std::vector<std::string> fCols;
   std::vector<std::size_t> fVecSizes;
   std::vector<RColumnLayout> fColumnLayout;
   std::size_t fBatchSize;
   std::size_t fSetSeed;

   // buffer quantities
   std::size_t fBufferCapacity;
   std::size_t fLowWatermark;

   std::size_t fTrainingClusterIdx{0};
   std::size_t fValidationClusterIdx{0};

   float fTestSize;

   std::unique_ptr<RDatasetLoader> fDatasetLoader;
   std::unique_ptr<RClusterLoader> fClusterLoader;
   std::unique_ptr<RBatchLoader> fTrainingBatchLoader;
   std::unique_ptr<RBatchLoader> fValidationBatchLoader;
   std::unique_ptr<RSampler> fTrainingSampler;
   std::unique_ptr<RSampler> fValidationSampler;

   std::unique_ptr<RFlat2DMatrixOperators> fTensorOperators;

   std::vector<ROOT::RDF::RNode> fRdfs;

   std::unique_ptr<std::thread> fLoadingThread;
   std::condition_variable fLoadingCondition;
   std::mutex fLoadingMutex;

   bool fDropRemainder;
   bool fShuffle;
   bool fLoadEager;
   std::string fSampleType;
   float fSampleRatio;
   bool fReplacement;

   bool fIsActive{false}; // Whether the loading thread is active

   bool fTrainingEpochActive{false};
   bool fValidationEpochActive{false};

   std::size_t fNumTrainingEntries;
   std::size_t fNumValidationEntries;

   // flattened buffers for chunks and temporary tensors (rows * cols)
   std::vector<RFlat2DMatrix> fTrainingDatasets;
   std::vector<RFlat2DMatrix> fValidationDatasets;

   RFlat2DMatrix fTrainingDataset;
   RFlat2DMatrix fValidationDataset;

   RFlat2DMatrix fSampledTrainingDataset;
   RFlat2DMatrix fSampledValidationDataset;

   std::size_t fTrainingEpochCount{0};
   std::size_t fValidationEpochCount{0};

   std::vector<RColumnLayout> MakeColumnLayout();

   /// \brief Opens a training or validation epoch and closes it again when done
   struct REpochGuard {
      RDataLoaderEngine &fEngine;
      bool fIsTraining;

      REpochGuard(RDataLoaderEngine &engine, bool isTraining);

      ~REpochGuard();
   };

public:
   RDataLoaderEngine(const std::vector<ROOT::RDF::RNode> &rdfs, const std::size_t batchSize,
                     const std::size_t batchesInMemory, const std::vector<std::string> &cols,
                     const std::vector<std::size_t> &vecSizes = {}, const float vecPadding = 0.0,
                     const float testSize = 0.0, bool shuffle = true, bool dropRemainder = true,
                     const std::size_t setSeed = 0, bool loadEager = false, std::string sampleType = "",
                     float sampleRatio = 1.0, bool replacement = false);

   ~RDataLoaderEngine();

   void DeActivate();

   void Activate();

   void Save(std::string_view dataset_name, std::string_view filename, bool isTraining, std::string_view outputFormat);

   void ActivateTrainingEpoch();

   void DeActivateTrainingEpoch();

   void ActivateValidationEpoch();

   void DeActivateValidationEpoch();

   void LoadData();

   void CreateTrainBatches();

   void CreateValidationBatches();

   RFlat2DMatrix GetTrainBatch();

   RFlat2DMatrix GetValidationBatch();

   std::size_t NumberOfTrainingBatches();
   std::size_t NumberOfValidationBatches();

   std::size_t TrainRemainderRows();
   std::size_t ValidationRemainderRows();

   bool IsActive();

   bool IsTrainingActive();

   bool IsValidationActive();
};

} // namespace ROOT::Experimental::Internal::ML

#endif // ROOT_INTERNAL_ML_RDATALOADERENGINE