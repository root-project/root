#ifndef TMVA_RBatchLoader
#define TMVA_RBatchLoader

#include <iostream>
#include <vector>
#include <memory>

// Imports for threading
#include <queue>
#include <mutex>
#include <condition_variable>

#include "TMVA/RTensor.hxx"
#include "TMVA/Tools.h"
#include "TRandom3.h"

namespace TMVA {
namespace Experimental {
namespace Internal {

class RBatchLoader {
private:
   std::size_t fBatchSize;
   std::size_t fNumColumns;
   std::size_t fMaxBatches;

   bool fIsActive = false;
   TMVA::RandomGenerator<TRandom3> fRng = TMVA::RandomGenerator<TRandom3>(0);

   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   std::queue<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatches;
   std::unique_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

   std::size_t fValidationIdx = 0;

   TMVA::Experimental::RTensor<float> fEmptyTensor = TMVA::Experimental::RTensor<float>({0});

public:
   RBatchLoader(const std::size_t batchSize, const std::size_t numColumns, const std::size_t maxBatches)
      : fBatchSize(batchSize), fNumColumns(numColumns), fMaxBatches(maxBatches)
   {
   }

   ~RBatchLoader() { DeActivate(); }

public:
   /// \brief Return a batch of data as a unique pointer.
   /// After the batch has been processed, it should be distroyed.
   /// \return Training batch
   const TMVA::Experimental::RTensor<float> &GetTrainBatch()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      fBatchCondition.wait(lock, [this]() { return !fTrainingBatchQueue.empty() || !fIsActive; });

      if (fTrainingBatchQueue.empty()) {
         fCurrentBatch = std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fTrainingBatchQueue.front());
      fTrainingBatchQueue.pop();

      fBatchCondition.notify_all();

      return *fCurrentBatch;
   }

   /// \brief Returns a batch of data for validation
   /// The owner of this batch has to be with the RBatchLoader.
   /// This is because the same validation batches should be used in all epochs.
   /// \return Validation batch
   const TMVA::Experimental::RTensor<float> &GetValidationBatch()
   {
      if (HasValidationData()) {
         return *fValidationBatches[fValidationIdx++].get();
      }

      return fEmptyTensor;
   }

   /// \brief Checks if there are more training batches available
   /// \return
   bool HasTrainData()
   {
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         if (!fTrainingBatchQueue.empty() || fIsActive)
            return true;
      }

      return false;
   }

   /// \brief Checks if there are more training batches available
   /// \return
   bool HasValidationData()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      return fValidationIdx < fValidationBatches.size();
   }

   /// \brief Activate the batchloader so it will accept chunks to batch
   void Activate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = true;
      }
      fBatchCondition.notify_all();
   }

   /// \brief DeActivate the batchloader. This means that no more batches are created.
   /// Batches can still be returned if they are already loaded
   void DeActivate()
   {
      {
         std::lock_guard<std::mutex> lock(fBatchLock);
         fIsActive = false;
      }
      fBatchCondition.notify_all();
   }

   /// \brief Create a batch filled with the events on the given idx
   /// \param chunkTensor
   /// \param idx
   /// \return
   std::unique_ptr<TMVA::Experimental::RTensor<float>>
   CreateBatch(const TMVA::Experimental::RTensor<float> &chunkTensor, const std::vector<std::size_t> idx)
   {
      auto batch =
         std::make_unique<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({fBatchSize, fNumColumns}));

      for (std::size_t i = 0; i < fBatchSize; i++) {
         std::copy(chunkTensor.GetData() + (idx[i] * fNumColumns), chunkTensor.GetData() + ((idx[i] + 1) * fNumColumns),
                   batch->GetData() + i * fNumColumns);
      }

      return batch;
   }

   /// \brief Create training batches from the given chunk of data based on the given event indices
   /// Batches are added to the training queue of batches
   /// The eventIndices can be shuffled to ensure random order for each epoch
   /// \param chunkTensor
   /// \param eventIndices
   /// \param shuffle
   void CreateTrainingBatches(const TMVA::Experimental::RTensor<float> &chunkTensor,
                              std::vector<std::size_t> eventIndices, const bool shuffle = true)
   {
      // Wait until less than a full chunk of batches are in the queue before loading splitting the next chunk into
      // batches
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         fBatchCondition.wait(lock, [this]() { return (fTrainingBatchQueue.size() < fMaxBatches) || !fIsActive; });
         if (!fIsActive)
            return;
      }

      if (shuffle)
         std::shuffle(eventIndices.begin(), eventIndices.end(), fRng); // Shuffle the order of idx

      std::vector<std::unique_ptr<TMVA::Experimental::RTensor<float>>> batches;

      // Create tasks of fBatchSize untill all idx are used
      for (std::size_t start = 0; (start + fBatchSize) <= eventIndices.size(); start += fBatchSize) {

         // Grab the first fBatchSize indices from the
         std::vector<std::size_t> idx;
         for (std::size_t i = start; i < (start + fBatchSize); i++) {
            idx.push_back(eventIndices[i]);
         }

         // Fill a batch
         batches.emplace_back(CreateBatch(chunkTensor, idx));
      }

      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         for (std::size_t i = 0; i < batches.size(); i++) {
            fTrainingBatchQueue.push(std::move(batches[i]));
         }
      }

      fBatchCondition.notify_one();
   }

   /// \brief Create validation batches from the given chunk based on the given event indices
   /// Batches are added to the vector of validation batches
   /// \param chunkTensor
   /// \param eventIndices
   void CreateValidationBatches(const TMVA::Experimental::RTensor<float> &chunkTensor,
                                const std::vector<std::size_t> eventIndices)
   {
      // Create tasks of fBatchSize untill all idx are used
      for (std::size_t start = 0; (start + fBatchSize) <= eventIndices.size(); start += fBatchSize) {

         std::vector<std::size_t> idx;

         for (std::size_t i = start; i < (start + fBatchSize); i++) {
            idx.push_back(eventIndices[i]);
         }

         {
            std::unique_lock<std::mutex> lock(fBatchLock);
            fValidationBatches.emplace_back(CreateBatch(chunkTensor, idx));
         }
      }
   }

   /// \brief Reset the validation process
   void StartValidation()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      fValidationIdx = 0;
   }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBatchLoader
