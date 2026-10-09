#include "ROOT/ML/RDataLoaderEngine.hxx"

namespace ROOT::Experimental::Internal::ML {

/// \brief Describe how the loader's columns map onto a batch-tensor row.
std::vector<RColumnLayout> RDataLoaderEngine::MakeColumnLayout()
{
   std::vector<RColumnLayout> layout;
   layout.reserve(fCols.size());

   std::size_t vecIdx = 0;
   std::size_t offset = 0;
   for (const auto &col : fCols) {
      const bool isVector = fRdfs[0].GetColumnType(col).find("RVec") != std::string::npos;
      const std::size_t width = isVector ? fVecSizes[vecIdx++] : 1;
      layout.push_back({col, offset, width, isVector});
      offset += width;
   }

   return layout;
}

RDataLoaderEngine::REpochGuard::REpochGuard(RDataLoaderEngine &engine, bool isTraining)
   : fEngine(engine), fIsTraining(isTraining)
{
   // Same order as the pythonization's epoch context managers
   fEngine.Activate();
   if (fIsTraining) {
      fEngine.CreateTrainBatches();
      fEngine.ActivateTrainingEpoch();
   } else {
      fEngine.CreateValidationBatches();
      fEngine.ActivateValidationEpoch();
   }
}

RDataLoaderEngine::REpochGuard::~REpochGuard()
{
   if (fIsTraining)
      fEngine.DeActivateTrainingEpoch();
   else
      fEngine.DeActivateValidationEpoch();
}

RDataLoaderEngine::RDataLoaderEngine(const std::vector<ROOT::RDF::RNode> &rdfs, const std::size_t batchSize,
                                     const std::size_t batchesInMemory, const std::vector<std::string> &cols,
                                     const std::vector<std::size_t> &vecSizes, const float vecPadding,
                                     const float testSize, bool shuffle, bool dropRemainder, const std::size_t setSeed,
                                     bool loadEager, std::string sampleType, float sampleRatio, bool replacement)
   : fRdfs(rdfs),
     fCols(cols),
     fVecSizes(vecSizes),
     fBatchSize(batchSize),
     fTestSize(testSize),
     fDropRemainder(dropRemainder),
     fSetSeed(setSeed),
     fShuffle(shuffle),
     fLoadEager(loadEager),
     fSampleType(sampleType),
     fSampleRatio(sampleRatio),
     fReplacement(replacement)
{
   fTensorOperators = std::make_unique<RFlat2DMatrixOperators>(fShuffle, fSetSeed);

   // describe the columns before the eager path releases the dataframes
   fColumnLayout = MakeColumnLayout();

   if (fLoadEager) {
      fDatasetLoader =
         std::make_unique<RDatasetLoader>(fRdfs, fTestSize, fCols, fVecSizes, vecPadding, fShuffle, fSetSeed);
      fDatasetLoader->SplitDatasets();

      if (fSampleType == "") {
         fDatasetLoader->ConcatenateDatasets();

         fTrainingDataset = fDatasetLoader->ReleaseTrainingDataset();
         fValidationDataset = fDatasetLoader->ReleaseValidationDataset();

         fNumTrainingEntries = fDatasetLoader->GetNumTrainingEntries();
         fNumValidationEntries = fDatasetLoader->GetNumValidationEntries();
      }

      else {
         fTrainingDatasets = fDatasetLoader->ReleaseTrainingDatasets();
         fValidationDatasets = fDatasetLoader->ReleaseValidationDatasets();

         fTrainingSampler =
            std::make_unique<RSampler>(fTrainingDatasets, fSampleType, fSampleRatio, fReplacement, fShuffle, fSetSeed);
         fValidationSampler = std::make_unique<RSampler>(fValidationDatasets, fSampleType, fSampleRatio, fReplacement,
                                                         fShuffle, fSetSeed);

         fNumTrainingEntries = fTrainingSampler->GetNumEntries();
         fNumValidationEntries = fValidationSampler->GetNumEntries();
      }

      // the dataset is in memory now, release the objects used to create it
      fDatasetLoader.reset();
      fRdfs.clear();
   }

   else {
      // scan cluster boundaries
      fClusterLoader =
         std::make_unique<RClusterLoader>(fRdfs, fCols, fVecSizes, vecPadding, fTestSize, fShuffle, fSetSeed);

      // derive buffer quantities
      fBufferCapacity = fBatchSize * batchesInMemory;
      // at least one batch, otherwise the refill threshold rounds down to 0 and nothing is ever loaded
      fLowWatermark = std::max(fBufferCapacity / 2, fBatchSize);

      // split cluster list into training and validation
      fClusterLoader->SplitDataset();
      fNumTrainingEntries = fClusterLoader->GetNumTrainingEntries();
      fNumValidationEntries = fClusterLoader->GetNumValidationEntries();
   }

   fTrainingBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fCols, fLoadingMutex, fLoadingCondition, fVecSizes,
                                                         fNumTrainingEntries, fDropRemainder);
   fValidationBatchLoader = std::make_unique<RBatchLoader>(fBatchSize, fCols, fLoadingMutex, fLoadingCondition,
                                                           fVecSizes, fNumValidationEntries, fDropRemainder);
}

RDataLoaderEngine::~RDataLoaderEngine()
{
   DeActivate();
}

void RDataLoaderEngine::DeActivate()
{
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      if (!fIsActive)
         return;
      fIsActive = false;
   }

   fLoadingCondition.notify_all();

   if (fLoadingThread) {
      if (fLoadingThread->joinable()) {
         fLoadingThread->join();
      }
   }

   fLoadingThread.reset();
}

/// \brief Activate the loading process by spawning the loading thread.
void RDataLoaderEngine::Activate()
{
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      if (fIsActive)
         return;

      fIsActive = true;
   }

   if (fLoadEager) {
      return;
   }

   fLoadingThread = std::make_unique<std::thread>(&RDataLoaderEngine::LoadData, this);
}

/// \brief Materialize one train/test split to disk by draining a full epoch through the normal batch
/// pipeline and Fill() each batch into \p filename instead of yielding it.
///
/// Filters, shuffling, the train/validation split and the batch_size/drop_remainder settings
/// are all inherited from the loader's configuration.
/// \param outputFormat Either "ttree" or "rntuple".
void RDataLoaderEngine::Save(std::string_view dataset_name, std::string_view filename, bool isTraining,
                             std::string_view outputFormat)
{
   // Cannot invoke mid-epoch
   if (isTraining ? IsTrainingActive() : IsValidationActive())
      throw std::runtime_error("RDataLoaderEngine::Save: this dataset is already being iterated elsewhere "
                               "(e.g. inside a training loop). Finish or stop that iteration before saving.");

   REpochGuard epoch(*this, isTraining);
   auto sink = CreateBatchSink(dataset_name, filename, fColumnLayout, outputFormat);

   while (true) {
      RFlat2DMatrix batch = isTraining ? GetTrainBatch() : GetValidationBatch();
      if (batch.GetSize() == 0)
         break;
      sink->FillBatch(batch);
   }

   sink->Commit();
}

/// \brief Activate the training epoch by starting the batchloader.
void RDataLoaderEngine::ActivateTrainingEpoch()
{
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      fTrainingEpochActive = true;
      fTrainingClusterIdx = 0;
      if (!fLoadEager) {
         // Shuffle the cluster indices at the beginning of each epoch
         fClusterLoader->ShuffleTrainingClusters(fTrainingEpochCount++);
      }
   }

   fTrainingBatchLoader->Activate();
   fLoadingCondition.notify_all();
}

void RDataLoaderEngine::DeActivateTrainingEpoch()
{
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      fTrainingEpochActive = false;
   }

   fTrainingBatchLoader->Reset();
   fTrainingBatchLoader->DeActivate();
   fLoadingCondition.notify_all();
}

void RDataLoaderEngine::ActivateValidationEpoch()
{
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      fValidationEpochActive = true;
      fValidationClusterIdx = 0;
      if (!fLoadEager) {
         fClusterLoader->ShuffleValidationClusters(fValidationEpochCount++);
      }
   }

   fValidationBatchLoader->Activate();
   fLoadingCondition.notify_all();
}

void RDataLoaderEngine::DeActivateValidationEpoch()
{
   {
      std::lock_guard<std::mutex> lock(fLoadingMutex);
      fValidationEpochActive = false;
   }

   fValidationBatchLoader->Reset();
   fValidationBatchLoader->DeActivate();
   fLoadingCondition.notify_all();
}

/// \brief Main loop for loading clusters and creating batches.
/// The producer (loading thread) will keep loading clusters and creating batches until the end of the epoch is
/// reached, or the generator is deactivated.
void RDataLoaderEngine::LoadData()
{
   std::unique_lock<std::mutex> lock(fLoadingMutex);

   while (true) {
      // Wait until we have work or shutdown
      fLoadingCondition.wait(lock, [&] {
         return !fIsActive ||
                (fTrainingEpochActive && fTrainingClusterIdx < fClusterLoader->GetNumTrainingClusters()) ||
                (fValidationEpochActive && fValidationClusterIdx < fClusterLoader->GetNumValidationClusters());
      });

      if (!fIsActive) {
         break;
      }

      // Helper: check if validation queue below watermark and needs the producer
      auto validationEmpty = [&] {
         if (!fValidationEpochActive || fValidationClusterIdx >= fClusterLoader->GetNumValidationClusters())
            return false;
         if (fValidationBatchLoader->isProducerDone())
            return false;
         return fValidationBatchLoader->GetNumBatchQueue() < fLowWatermark / fBatchSize;
      };

      // -- TRAINING --
      if (fTrainingEpochActive) {
         const std::size_t numTrainingClusters = fClusterLoader->GetNumTrainingClusters();

         while (true) {
            // Stop conditions (shutdown or epoch end)
            if (!fIsActive || !fTrainingEpochActive)
               break;

            // No more chunks to load: signal consumers
            if (fTrainingClusterIdx >= numTrainingClusters) {
               fTrainingBatchLoader->MarkProducerDone();
               break;
            }

            // In the case of training prefetching, we could start requesting data for the next training loop while
            // validation is active and might need data. To avoid getting stuck in the training loop, we check if the
            // validation queue is below watermark and if so, we break out of the training loop.
            if (validationEmpty()) {
               break;
            }

            // If queue is not empty, wait until it drains below watermark, or validation needs data, or we are
            // deactivated.
            if (fTrainingBatchLoader->GetNumBatchQueue() >= fLowWatermark / fBatchSize) {
               fLoadingCondition.wait(lock, [&] {
                  return !fIsActive || !fTrainingEpochActive ||
                         fTrainingBatchLoader->GetNumBatchQueue() < (fLowWatermark / fBatchSize) || validationEmpty();
               });
               continue;
            }

            // Accumulate clusters to load, enough to fill the buffer, or until we run out of clusters
            std::vector<RClusterRange> trainClustersToLoad;
            auto accumulatedEntries = 0;
            const bool discovering = !fClusterLoader->IsSplitDiscovered();
            while (fTrainingClusterIdx < numTrainingClusters && accumulatedEntries < fBufferCapacity &&
                   (!discovering || trainClustersToLoad.empty())) {
               const auto &cluster = fClusterLoader->GetTrainingClusters()[fTrainingClusterIdx++];
               trainClustersToLoad.push_back(cluster);
               accumulatedEntries += cluster.GetNumEntries();
            }

            const bool isLastBuffer = (fTrainingClusterIdx >= numTrainingClusters);

            // Release lock while reading and loading data to allow the consumer to access the queue freely in
            // parallel. The loading thread re-acquires the lock in CreateBatches when it needs to push batches to
            // the queue.
            lock.unlock();
            RFlat2DMatrix stagingBuffer(accumulatedEntries, fClusterLoader->GetNumChunkCols());
            std::size_t rowOffset = 0;

            for (auto &cluster : trainClustersToLoad) {
               auto loadedEntries = fClusterLoader->LoadTrainingClusterInto(stagingBuffer, cluster.rdfIdx,
                                                                            cluster.start, cluster.end, rowOffset);
               if (discovering) {
                  // For the first epoch, we might discover that the cluster has fewer entries than expected because
                  // of filters
                  cluster.SetNumEntries(loadedEntries);
               }
               rowOffset += cluster.GetNumEntries();
            }

            if (discovering && fNumTrainingEntries == 0 && fClusterLoader->GetNumTrainingEntries() > 0) {
               fNumTrainingEntries = fClusterLoader->GetNumTrainingEntries();
               fNumValidationEntries = fClusterLoader->GetNumValidationEntries();
               fTrainingBatchLoader->RecalculateBatchCounts(fNumTrainingEntries);
               fValidationBatchLoader->RecalculateBatchCounts(fNumValidationEntries);
            }

            if (rowOffset < static_cast<std::size_t>(accumulatedEntries)) {
               stagingBuffer.Resize(rowOffset, stagingBuffer.GetCols());
            }

            RFlat2DMatrix shuffledStagingBuffer;
            fTrainingBatchLoader->CreateBatches(fTensorOperators->ShuffleTensor(shuffledStagingBuffer, stagingBuffer),
                                                isLastBuffer);

            // Re-acquire the lock before the next iteration to check conditions and update indices
            lock.lock();

            if (isLastBuffer && discovering) {
               fClusterLoader->FinaliseSplitDiscovery();
            }
         }
      }

      // -- VALIDATION --
      if (fValidationEpochActive) {
         const std::size_t numValidationClusters = fClusterLoader->GetNumValidationClusters();

         while (true) {
            // Stop conditions (shutdown or epoch end)
            if (!fIsActive || !fValidationEpochActive)
               break;

            // No more chunks to load: signal consumers
            if (fValidationClusterIdx >= numValidationClusters) {
               fValidationBatchLoader->MarkProducerDone();
               break;
            }

            // If queue is not hungry, wait until it drains below watermark, or we are deactivated
            if (fValidationBatchLoader->GetNumBatchQueue() >= (fLowWatermark / fBatchSize)) {
               fLoadingCondition.wait(lock, [&] {
                  return !fIsActive || !fValidationEpochActive ||
                         fValidationBatchLoader->GetNumBatchQueue() < (fLowWatermark / fBatchSize);
               });
               continue;
            }

            // Accumulate clusters to load, enough to fill the buffer, or until we run out of clusters
            std::vector<RClusterRange> valClustersToLoad;
            auto accumulatedEntries = 0;
            while (fValidationClusterIdx < numValidationClusters && accumulatedEntries < fBufferCapacity) {
               const auto &cluster = fClusterLoader->GetValidationClusters()[fValidationClusterIdx++];
               valClustersToLoad.push_back(cluster);
               accumulatedEntries += cluster.GetNumEntries();
            }

            const bool isLastBuffer = (fValidationClusterIdx >= numValidationClusters);

            lock.unlock();

            RFlat2DMatrix stagingBuffer(accumulatedEntries, fClusterLoader->GetNumChunkCols());
            std::size_t rowOffset = 0;

            for (const auto &cluster : valClustersToLoad) {
               fClusterLoader->LoadValidationClusterInto(stagingBuffer, cluster.rdfIdx, cluster.start, cluster.end,
                                                         rowOffset);
               rowOffset += cluster.GetNumEntries();
            }

            RFlat2DMatrix shuffledStagingBuffer;
            fValidationBatchLoader->CreateBatches(fTensorOperators->ShuffleTensor(shuffledStagingBuffer, stagingBuffer),
                                                  isLastBuffer);

            lock.lock();
         }
      }
   }
}

/// \brief Create training batches by first loading a chunk (see RClusterLoader) and split it into batches (see
/// RBatchLoader)
void RDataLoaderEngine::CreateTrainBatches()
{
   fTrainingBatchLoader->Activate();

   if (fLoadEager) {
      RFlat2DMatrix *source = &fSampledTrainingDataset;
      if (fSampleType == "") {
         source = &fTensorOperators->ShuffleTensor(fSampledTrainingDataset, fTrainingDataset);
      }

      else {
         fTrainingSampler->Sampler(fSampledTrainingDataset);
      }

      fTrainingBatchLoader->CreateBatches(*source, true);
      fTrainingBatchLoader->MarkProducerDone();
   }
}

/// \brief Creates validation batches by first loading a chunk (see RClusterLoader), and then split it into batches
/// (see RBatchLoader)
void RDataLoaderEngine::CreateValidationBatches()
{
   fValidationBatchLoader->Activate();

   if (fLoadEager) {
      RFlat2DMatrix *source = &fSampledValidationDataset;
      if (fSampleType == "") {
         source = &fTensorOperators->ShuffleTensor(fSampledValidationDataset, fValidationDataset);
      }

      else {
         fValidationSampler->Sampler(fSampledValidationDataset);
      }

      fValidationBatchLoader->CreateBatches(*source, true);
      fValidationBatchLoader->MarkProducerDone();
   }
}

/// \brief Loads a training batch from the queue
RFlat2DMatrix RDataLoaderEngine::GetTrainBatch()
{
   // Get next batch if available
   return fTrainingBatchLoader->GetBatch();
}

/// \brief Loads a validation batch from the queue
RFlat2DMatrix RDataLoaderEngine::GetValidationBatch()
{
   // Get next batch if available
   return fValidationBatchLoader->GetBatch();
}

std::size_t RDataLoaderEngine::NumberOfTrainingBatches()
{
   return fTrainingBatchLoader->GetNumBatches();
}

std::size_t RDataLoaderEngine::NumberOfValidationBatches()
{
   return fValidationBatchLoader->GetNumBatches();
}

std::size_t RDataLoaderEngine::TrainRemainderRows()
{
   return fTrainingBatchLoader->GetNumRemainderRows();
}

std::size_t RDataLoaderEngine::ValidationRemainderRows()
{
   return fValidationBatchLoader->GetNumRemainderRows();
}

bool RDataLoaderEngine::IsActive()
{
   std::lock_guard<std::mutex> lock(fLoadingMutex);
   return fIsActive;
}

bool RDataLoaderEngine::IsTrainingActive()
{
   std::lock_guard<std::mutex> lock(fLoadingMutex);
   return fTrainingEpochActive;
}

bool RDataLoaderEngine::IsValidationActive()
{
   std::lock_guard<std::mutex> lock(fLoadingMutex);
   return fValidationEpochActive;
}

} // namespace ROOT::Experimental::Internal::ML
