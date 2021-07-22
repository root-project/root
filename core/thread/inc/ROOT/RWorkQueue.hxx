/// \file ROOT/RWorkQueue.hxx
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2021-07-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RWorkQueue
#define ROOT_RWorkQueue

#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace ROOT {
namespace Internal {

/// A thread-safe queue of Ts. The items are moved in and out of the queue
/// and are owned by the queue for their retention time in the queue.
/// FIFO semantics.
/// The work queue can connect (multiple) producers to (multiple) consumers.
/// The queue is bounded: when full, producers block and when empty, consumers
/// block.
template <typename T>
class RWorkQueue {
private:
   /// Maximum number of Ts in the queue. Pushing further will block until items are consumed.
   std::size_t fLimit;
   /// The queue storage
   std::deque<T> fQueue;
   /// Protects all internal state
   std::mutex fLock;
   /// Signals if there are items enqueued
   std::condition_variable fCvHasItems;
   /// Signals if there is space to enqueue more items
   std::condition_variable fCvHasSpace;

public:
   explicit RWorkQueue(std::size_t limit) : fLimit(limit) {}
   RWorkQueue(const RWorkQueue &other) = delete;
   RWorkQueue(RWorkQueue &&other) = default;
   RWorkQueue &operator =(const RWorkQueue &other) = delete;
   RWorkQueue &operator =(RWorkQueue &other) = default;
   ~RWorkQueue() = default;

   /// Pushes a new item into the queue, blocks as long as queue is full
   void Enqueue(T &&item)
   {
      std::unique_lock<std::mutex> lock(fLock);
      fCvHasSpace.wait(lock, [&]{ return fQueue.size() < fLimit; });

      bool wasEmpty = fQueue.empty();
      fQueue.emplace_back(std::forward<T>(item));
      if (wasEmpty)
         fCvHasItems.notify_all();
   }

   /// Retrieves the oldest item in the queue, blocks as long as queue is empty
   T Pop()
   {
      std::unique_lock<std::mutex> lock(fLock);
      fCvHasItems.wait(lock, [&]{ return !fQueue.empty(); });

      bool wasFull = fQueue.size() == fLimit;
      auto item = std::move(fQueue.front());
      fQueue.pop_front();

      if (wasFull)
         fCvHasSpace.notify_all();

      return item;
   }

   bool IsEmpty()
   {
      std::lock_guard<std::mutex> guard(fLock);
      return fQueue.empty();
   }

   bool IsFull()
   {
      std::lock_guard<std::mutex> guard(fLock);
      return fQueue.size() == fLimit;
   }

   std::size_t GetSize()
   {
      std::lock_guard<std::mutex> guard(fLock);
      return fQueue.size();
   }
}; // class RWorkQueue<T>

} // namespace Internal
} // namespace ROOT

#endif
