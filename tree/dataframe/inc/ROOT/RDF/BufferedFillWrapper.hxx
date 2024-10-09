// Author: Stephan Hageboeck CERN  10/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_BUFFERED_FILL
#define ROOT_BUFFERED_FILL

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace ROOT::Internal::RDF {

/**
 * Wrapper for any kind of thread-unsafe object that has a Fill function.
 *
 * This wrapper uses thread-safe queues to protect the Fill function of the underlying object.
 * Instead of having to clone the object, a single instance can be used, saving lots or RAM.
 * The use of multiple fill queues enables other threads to make progress while the managed
 * object is being filled under lock.
 *
 * \tparam Payload_t Object with a fill function.
 * \tparam FillArgs Argument type(s) of the fill function.
 */
template <typename Payload_t, typename... FillArgs>
class BufferedFillWrapper {
   using Tuple_t = std::tuple<FillArgs...>;
   /// This struct represents all data that is needed to queue up fills.
   /// We align to a typical cache line to avoid false sharing.
   struct alignas(64) FillQueue {
      std::vector<Tuple_t> data;
      std::atomic_uint requestedSlots = 0;
      std::atomic_uint refCount = 0;
      std::atomic_bool acceptsNewData = true;
   };

   std::array<FillQueue, 2> fQueues;
   std::atomic<FillQueue *> fActiveQueue;

   std::shared_ptr<Payload_t> fManagedObject;
   std::mutex fManagedObjectMutex;

public:
   class LockedHandle {
      std::unique_lock<std::mutex> fLock;
      std::shared_ptr<Payload_t> fPayload;

   public:
      LockedHandle(std::mutex &mutex, std::shared_ptr<Payload_t> payload) : fLock{mutex}, fPayload{std::move(payload)}
      {
      }

      Payload_t *operator->() const { return fPayload.get(); }
   };

   /// @brief Create a thread-safe wrapper around an object to be filled.
   /// @param objectToFill Object to fill, e.g. TH3D or similar.
   /// @param queueSize Size of the fill buffers in case the object is unavailable.
   /// If no value is passed, the queue will be set to 4K.
   BufferedFillWrapper(std::shared_ptr<Payload_t> objectToFill, unsigned int queueSize = 0)
      : fActiveQueue(fQueues.begin()), fManagedObject(std::move(objectToFill))
   {
      if (fManagedObject == nullptr)
         throw std::invalid_argument{"BufferedFillWrapper: No object to fill provided."};

      if (queueSize == 0) {
         queueSize = std::max(32ul, 4096 / sizeof(Tuple_t));
      }
      for (auto &queue : fQueues) {
         queue.data.resize(queueSize);
         queue.requestedSlots = 0;
      }
   }
   BufferedFillWrapper(const BufferedFillWrapper &) = delete;
   BufferedFillWrapper(BufferedFillWrapper &&) = default;

   struct RefCountGuard {
      RefCountGuard(FillQueue &slotData) : fSlotData{slotData}
      {
         fSlotData.refCount.fetch_add(1, std::memory_order_release);
      }
      ~RefCountGuard() { fSlotData.refCount.fetch_sub(1, std::memory_order_release); }
      FillQueue &fSlotData;
   };

   /// @brief Fill the managed object under a lock, or (if the lock is unavailable) place the arguments
   /// into a buffer for later filling.
   /// @param ...args Arguments of the fill function.
   void Fill(FillArgs... args)
   {
      bool dataFilled = false;
      do {
         FillQueue *const activeQueue = fActiveQueue.load();
         if (activeQueue == nullptr) {
            throw std::logic_error("BufferedFillWrapper::Fill() called after the object has been retrieved.");
         }

         if (std::unique_lock objectLock{fManagedObjectMutex, std::try_to_lock}; objectLock.owns_lock()) {
            FillQueue *queueToFlush = nullptr;
            if (activeQueue->requestedSlots > 0)
               queueToFlush = &CycleQueues();

            fManagedObject->Fill(args...);
            dataFilled = true;

            if (queueToFlush)
               FlushQueue(*queueToFlush);
         } else {
            RefCountGuard refCountGuard{*activeQueue};

            if (!activeQueue->acceptsNewData)
               continue;

            const auto ourSlot = activeQueue->requestedSlots.fetch_add(1);
            if (ourSlot < activeQueue->data.size()) {
               activeQueue->data[ourSlot] = Tuple_t{args...};
               dataFilled = true;
            }
         }
      } while (!dataFilled);
   }

   /// @brief Lock the underlying object and return a handle to it.
   /// All fill queues are flushed before the object is returned, and the lock is held
   /// as long as the LockedHandle is alive, so Fill threads cannot make progress.
   /// @return Handle to the object. While the handle is alive, the object is locked.
   LockedHandle Get()
   {
      LockedHandle handle{fManagedObjectMutex, fManagedObject};

      for (auto &queue : fQueues) {
         FlushQueue(queue);
      }

      return std::move(handle);
   }

   /// Flush all queues, and release the underlying object.
   std::shared_ptr<Payload_t> Release()
   {
      fActiveQueue = nullptr;
      std::scoped_lock lock{fManagedObjectMutex};

      for (auto &queue : fQueues) {
         FlushQueue(queue);
      }

      auto ret = fManagedObject;
      fManagedObject = nullptr;

      return ret;
   }

private:
   /// @brief Move over to the next fill queue.
   /// Returns the queue that should be flushed.
   FillQueue &CycleQueues()
   {
      auto activeQueue = fActiveQueue.load();

      auto it = fQueues.begin();
      while (it != activeQueue)
         ++it;
      fActiveQueue = (++it) == fQueues.end() ? fQueues.begin() : it;

      activeQueue->acceptsNewData = false;
      return *activeQueue;
   }

   /// @brief Flush a queue by filling all items into the managed object.
   /// @param queueToFlush Queue of arguments to be filled into the managed object.
   void FlushQueue(FillQueue &queueToFlush)
   {
      queueToFlush.acceptsNewData = false;

      if (!fManagedObject)
         throw std::logic_error{"BufferedFillWrapper: The managed object has already been retrieved."};

      // Wait until all write operations have completed
      while (queueToFlush.refCount.load(std::memory_order_acquire) > 0)
         ;

      const auto nFilled =
         std::min<unsigned int>(queueToFlush.data.size(), queueToFlush.requestedSlots.load(std::memory_order_acquire));

      auto &toFill = *fManagedObject;
      auto callFill = [&toFill](FillArgs... args) { toFill.Fill(args...); };

      for (unsigned int i = 0; i < nFilled; ++i) {
         std::apply(callFill, queueToFlush.data[i]);
      }

      queueToFlush.requestedSlots = 0;
      queueToFlush.acceptsNewData = true;
   }
};

} // namespace ROOT::Internal::RDF

#endif // ROOT_BUFFERED_FILL
