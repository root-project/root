#ifndef TMVA_RBatchLoader
#define TMVA_RBatchLoader

#include <vector>
#include <memory>
#include <numeric>

// Imports for threading
#include <queue>
#include <mutex>
#include <condition_variable>

#include "TMVA/RTensor.hxx"
#include "TMVA/Tools.h"

namespace TMVA {
namespace Experimental {
namespace Internal {

class RBatchLoader {
private:
   const TMVA::Experimental::RTensor<float> & fChunkTensor;
   std::size_t fBatchSize;
   std::size_t fNumColumns;
   std::size_t fMaxBatches;

   bool fIsActive = false;

   std::mutex fBatchLock;
   std::condition_variable fBatchCondition;

   std::queue<std::shared_ptr<TMVA::Experimental::RTensor<float>>> fTrainingBatchQueue;
   std::queue<std::shared_ptr<TMVA::Experimental::RTensor<float>>> fValidationBatchQueue;
   std::shared_ptr<TMVA::Experimental::RTensor<float>> fCurrentBatch;

public:
   RBatchLoader(const TMVA::Experimental::RTensor<float> & chunkTensor, const std::size_t batchSize,
               const std::size_t numColumns, const std::size_t maxBatches)
      : fChunkTensor(chunkTensor),
        fBatchSize(batchSize),
        fNumColumns(numColumns),
        fMaxBatches(maxBatches)
   {
   }

   ~RBatchLoader() { DeActivate(); }

public:
   /// \brief Return a batch of data as a unique pointer.
   /// After the batch has been processed, it should be destroyed.
   /// \return Training batch
   const TMVA::Experimental::RTensor<float> &GetTrainBatch()
   {
      std::unique_lock<std::mutex> lock(fBatchLock);
      fBatchCondition.wait(lock, [this]() {return !fTrainingBatchQueue.empty() || !fIsActive; });

      if (fTrainingBatchQueue.empty()) {
         fCurrentBatch = std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
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
      if (fValidationBatchQueue.empty()) {
         fCurrentBatch = std::make_shared<TMVA::Experimental::RTensor<float>>(std::vector<std::size_t>({0}));
         return *fCurrentBatch;
      }

      fCurrentBatch = std::move(fValidationBatchQueue.front());
      fValidationBatchQueue.pop();

      return *fCurrentBatch;
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

   void UnloadTrainingVectors(std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & batches)
   {
      // Wait until less than a full chunk of batches are in the queue before loading splitting the next chunk into
      // batches
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         fBatchCondition.wait(lock, [this]() { return (fTrainingBatchQueue.size() < fMaxBatches) || !fIsActive; });
         if (!fIsActive)
            return;
      }

      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         for (std::size_t i = 0; i < batches.size(); i++) {
            fTrainingBatchQueue.push(std::move(batches[i]));
         }
      }

      fBatchCondition.notify_all();
   }

   void UnloadValidationVectors(std::vector<std::shared_ptr<TMVA::Experimental::RTensor<float>>> & batches)
   {
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         for (std::size_t i = 0; i < batches.size(); i++) {
            fValidationBatchQueue.push(std::move(batches[i]));
         }
      }
   }

   void UnloadRemainder(std::pair<std::shared_ptr<TMVA::Experimental::RTensor<float>>,std::shared_ptr<TMVA::Experimental::RTensor<float>>> remainders){
      {
         std::unique_lock<std::mutex> lock(fBatchLock);
         fTrainingBatchQueue.push(std::move(remainders.first));
      }

      fValidationBatchQueue.push(remainders.second);
   }
};

} // namespace Internal
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_RBatchLoader
