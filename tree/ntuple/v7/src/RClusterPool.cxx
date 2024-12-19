/// \file RClusterPool.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2020-03-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RClusterPool.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TError.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <utility>

bool ROOT::Experimental::Internal::RClusterPool::RInFlightCluster::operator<(const RInFlightCluster &other) const
{
   if (fClusterKey.fClusterId == other.fClusterKey.fClusterId) {
      if (fClusterKey.fPhysicalColumnSet.size() == other.fClusterKey.fPhysicalColumnSet.size()) {
         for (auto itr1 = fClusterKey.fPhysicalColumnSet.begin(), itr2 = other.fClusterKey.fPhysicalColumnSet.begin();
              itr1 != fClusterKey.fPhysicalColumnSet.end(); ++itr1, ++itr2) {
            if (*itr1 == *itr2)
               continue;
            return *itr1 < *itr2;
         }
         // *this == other
         return false;
      }
      return fClusterKey.fPhysicalColumnSet.size() < other.fClusterKey.fPhysicalColumnSet.size();
   }
   return fClusterKey.fClusterId < other.fClusterKey.fClusterId;
}

ROOT::Experimental::Internal::RClusterPool::RClusterPool(RPageSource &pageSource, unsigned int clusterBunchSize)
   : fPageSource(pageSource),
     fClusterBunchSize(clusterBunchSize),
     fPool(2 * clusterBunchSize),
     fThreadIo(&RClusterPool::ExecReadClusters, this)
{
   R__ASSERT(clusterBunchSize > 0);
}

ROOT::Experimental::Internal::RClusterPool::~RClusterPool()
{
   {
      // Controlled shutdown of the I/O thread
      std::unique_lock<std::mutex> lock(fLockWorkQueue);
      fReadQueue.emplace_back(RReadItem());
      fCvHasReadWork.notify_one();
   }
   fThreadIo.join();
}

void ROOT::Experimental::Internal::RClusterPool::ExecReadClusters()
{
   std::deque<RReadItem> readItems;
   while (true) {
      {
         std::unique_lock<std::mutex> lock(fLockWorkQueue);
         fCvHasReadWork.wait(lock, [&]{ return !fReadQueue.empty(); });
         std::swap(readItems, fReadQueue);
      }

      while (!readItems.empty()) {
         std::vector<RCluster::RKey> clusterKeys;
         std::int64_t bunchId = -1;
         for (unsigned i = 0; i < readItems.size(); ++i) {
            const auto &item = readItems[i];
            // `kInvalidDescriptorId` is used as a marker for thread cancellation. Such item causes the
            // thread to terminate; thus, it must appear last in the queue.
            if (R__unlikely(item.fClusterKey.fClusterId == kInvalidDescriptorId)) {
               R__ASSERT(i == (readItems.size() - 1));
               return;
            }
            if ((bunchId >= 0) && (item.fBunchId != bunchId))
               break;
            bunchId = item.fBunchId;
            clusterKeys.emplace_back(item.fClusterKey);
         }

         auto clusters = fPageSource.LoadClusters(clusterKeys);
         for (std::size_t i = 0; i < clusters.size(); ++i) {
            readItems[i].fPromise.set_value(std::move(clusters[i]));
         }
         readItems.erase(readItems.begin(), readItems.begin() + clusters.size());
      }
   } // while (true)
}

ROOT::Experimental::Internal::RCluster *
ROOT::Experimental::Internal::RClusterPool::FindInPool(DescriptorId_t clusterId) const
{
   for (const auto &cptr : fPool) {
      if (cptr && (cptr->GetId() == clusterId))
         return cptr.get();
   }
   return nullptr;
}

size_t ROOT::Experimental::Internal::RClusterPool::FindFreeSlot() const
{
   auto N = fPool.size();
   for (unsigned i = 0; i < N; ++i) {
      if (!fPool[i])
         return i;
   }

   R__ASSERT(false);
   return N;
}


namespace {

/// Helper class for the (cluster, column list) pairs that should be loaded in the background
class RProvides {
   using DescriptorId_t = ROOT::Experimental::DescriptorId_t;
   using ColumnSet_t = ROOT::Experimental::Internal::RCluster::ColumnSet_t;

public:
   struct RInfo {
      std::int64_t fBunchId = -1;
      std::int64_t fFlags = 0;
      ColumnSet_t fPhysicalColumnSet;
   };

   static constexpr std::int64_t kFlagRequired = 0x01;
   static constexpr std::int64_t kFlagLast     = 0x02;

private:
   std::map<DescriptorId_t, RInfo> fMap;

public:
   void Insert(DescriptorId_t clusterId, const RInfo &info)
   {
      fMap.emplace(clusterId, info);
   }

   bool Contains(DescriptorId_t clusterId) {
      return fMap.count(clusterId) > 0;
   }

   std::size_t GetSize() const { return fMap.size(); }

   void Erase(DescriptorId_t clusterId, const ColumnSet_t &physicalColumns)
   {
      auto itr = fMap.find(clusterId);
      if (itr == fMap.end())
         return;
      ColumnSet_t d;
      std::copy_if(itr->second.fPhysicalColumnSet.begin(), itr->second.fPhysicalColumnSet.end(),
                   std::inserter(d, d.end()),
                   [&physicalColumns](DescriptorId_t needle) { return physicalColumns.count(needle) == 0; });
      if (d.empty()) {
         fMap.erase(itr);
      } else {
         itr->second.fPhysicalColumnSet = d;
      }
   }

   decltype(fMap)::iterator begin() { return fMap.begin(); }
   decltype(fMap)::iterator end() { return fMap.end(); }
};

} // anonymous namespace

ROOT::Experimental::Internal::RCluster *
ROOT::Experimental::Internal::RClusterPool::GetCluster(DescriptorId_t clusterId,
                                                       const RCluster::ColumnSet_t &physicalColumns)
{
   std::set<DescriptorId_t> keep;
   RProvides provide;
   {
      auto descriptorGuard = fPageSource.GetSharedDescriptorGuard();

      // Determine previous cluster ids that we keep if they happen to be in the pool
      auto prev = clusterId;
      for (unsigned int i = 0; i < fWindowPre; ++i) {
         prev = descriptorGuard->FindPrevClusterId(prev);
         if (prev == kInvalidDescriptorId)
            break;
         keep.insert(prev);
      }

      // Determine following cluster ids and the column ids that we want to make available
      RProvides::RInfo provideInfo;
      provideInfo.fPhysicalColumnSet = physicalColumns;
      provideInfo.fBunchId = fBunchId;
      provideInfo.fFlags = RProvides::kFlagRequired;
      for (DescriptorId_t i = 0, next = clusterId; i < 2 * fClusterBunchSize; ++i) {
         if (i == fClusterBunchSize)
            provideInfo.fBunchId = ++fBunchId;

         auto cid = next;
         next = descriptorGuard->FindNextClusterId(cid);
         if (next != kInvalidNTupleIndex) {
            if (!fPageSource.GetEntryRange().IntersectsWith(descriptorGuard->GetClusterDescriptor(next)))
               next = kInvalidNTupleIndex;
         }
         if (next == kInvalidDescriptorId)
            provideInfo.fFlags |= RProvides::kFlagLast;

         provide.Insert(cid, provideInfo);

         if (next == kInvalidDescriptorId)
            break;
         provideInfo.fFlags = 0;
      }
   } // descriptorGuard

   // Clear the cache from clusters not the in the look-ahead or the look-back window
   for (auto &cptr : fPool) {
      if (!cptr)
         continue;
      if (provide.Contains(cptr->GetId()) > 0)
         continue;
      if (keep.count(cptr->GetId()) > 0)
         continue;
      cptr.reset();
   }

   // Move clusters that meanwhile arrived into cache pool
   {
      // This lock is held during iteration over several data structures: the collection of in-flight clusters,
      // the current pool of cached clusters, and the set of cluster ids to be preloaded.
      // All three collections are expected to be small (certainly < 100, more likely < 10).  All operations
      // are non-blocking and moving around small items (pointers, ids, etc).  Thus the overall locking time should
      // still be reasonably small and the lock is rarely taken (usually once per cluster).
      std::lock_guard<std::mutex> lockGuard(fLockWorkQueue);

      for (auto itr = fInFlightClusters.begin(); itr != fInFlightClusters.end(); ) {
         R__ASSERT(itr->fFuture.valid());
         if (itr->fFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            // Remove the set of columns that are already scheduled for being loaded
            provide.Erase(itr->fClusterKey.fClusterId, itr->fClusterKey.fPhysicalColumnSet);
            ++itr;
            continue;
         }

         auto cptr = itr->fFuture.get();
         R__ASSERT(cptr);

         const bool isExpired =
            !provide.Contains(itr->fClusterKey.fClusterId) && (keep.count(itr->fClusterKey.fClusterId) == 0);
         if (isExpired) {
            cptr.reset();
            itr = fInFlightClusters.erase(itr);
            continue;
         }

         // Noop unless the page source has a task scheduler
         fPageSource.UnzipCluster(cptr.get());

         // We either put a fresh cluster into a free slot or we merge the cluster with an existing one
         auto existingCluster = FindInPool(cptr->GetId());
         if (existingCluster) {
            existingCluster->Adopt(std::move(*cptr));
         } else {
            auto idxFreeSlot = FindFreeSlot();
            fPool[idxFreeSlot] = std::move(cptr);
         }
         itr = fInFlightClusters.erase(itr);
      }

      // Determine clusters which get triggered for background loading
      for (auto &cptr : fPool) {
         if (!cptr)
            continue;
         provide.Erase(cptr->GetId(), cptr->GetAvailPhysicalColumns());
      }

      // Figure out if enough work accumulated to justify I/O calls
      bool skipPrefetch = false;
      if (provide.GetSize() < fClusterBunchSize) {
         skipPrefetch = true;
         for (const auto &kv : provide) {
            if ((kv.second.fFlags & (RProvides::kFlagRequired | RProvides::kFlagLast)) == 0)
               continue;
            skipPrefetch = false;
            break;
         }
      }

      // Update the work queue and the in-flight cluster list with new requests. We already hold the work queue
      // mutex
      // TODO(jblomer): we should ensure that clusterId is given first to the I/O thread.  That is usually the
      // case but it's not ensured by the code
      if (!skipPrefetch) {
         for (const auto &kv : provide) {
            R__ASSERT(!kv.second.fPhysicalColumnSet.empty());

            RReadItem readItem;
            readItem.fClusterKey.fClusterId = kv.first;
            readItem.fBunchId = kv.second.fBunchId;
            readItem.fClusterKey.fPhysicalColumnSet = kv.second.fPhysicalColumnSet;

            RInFlightCluster inFlightCluster;
            inFlightCluster.fClusterKey.fClusterId = kv.first;
            inFlightCluster.fClusterKey.fPhysicalColumnSet = kv.second.fPhysicalColumnSet;
            inFlightCluster.fFuture = readItem.fPromise.get_future();
            fInFlightClusters.emplace_back(std::move(inFlightCluster));

            fReadQueue.emplace_back(std::move(readItem));
         }
         if (!fReadQueue.empty())
            fCvHasReadWork.notify_one();
      }
   } // work queue lock guard

   return WaitFor(clusterId, physicalColumns);
}

ROOT::Experimental::Internal::RCluster *
ROOT::Experimental::Internal::RClusterPool::WaitFor(DescriptorId_t clusterId,
                                                    const RCluster::ColumnSet_t &physicalColumns)
{
   while (true) {
      // Fast exit: the cluster happens to be already present in the cache pool
      auto result = FindInPool(clusterId);
      if (result) {
         bool hasMissingColumn = false;
         for (auto cid : physicalColumns) {
            if (result->ContainsColumn(cid))
               continue;

            hasMissingColumn = true;
            break;
         }
         if (!hasMissingColumn)
            return result;
      }

      // Otherwise the missing data must have been triggered for loading by now, so block and wait
      decltype(fInFlightClusters)::iterator itr;
      {
         std::lock_guard<std::mutex> lockGuardInFlightClusters(fLockWorkQueue);
         itr = fInFlightClusters.begin();
         for (; itr != fInFlightClusters.end(); ++itr) {
            if (itr->fClusterKey.fClusterId == clusterId)
               break;
         }
         R__ASSERT(itr != fInFlightClusters.end());
         // Note that the fInFlightClusters is accessed concurrently only by the I/O thread.  The I/O thread
         // never changes the structure of the in-flight clusters array (it does not add, remove, or swap elements).
         // Therefore, it is safe to access the element pointed to by itr here even after fLockWorkQueue
         // is released.  We need to release the lock before potentially blocking on the cluster future.
      }

      auto cptr = itr->fFuture.get();
      // We were blocked waiting for the cluster, so assume that nobody discarded it.
      R__ASSERT(cptr != nullptr);

      // Noop unless the page source has a task scheduler
      fPageSource.UnzipCluster(cptr.get());

      if (result) {
         result->Adopt(std::move(*cptr));
      } else {
         auto idxFreeSlot = FindFreeSlot();
         fPool[idxFreeSlot] = std::move(cptr);
      }

      std::lock_guard<std::mutex> lockGuardInFlightClusters(fLockWorkQueue);
      fInFlightClusters.erase(itr);
   }
}

void ROOT::Experimental::Internal::RClusterPool::WaitForInFlightClusters()
{
   while (true) {
      decltype(fInFlightClusters)::iterator itr;
      {
         std::lock_guard<std::mutex> lockGuardInFlightClusters(fLockWorkQueue);
         itr = fInFlightClusters.begin();
         if (itr == fInFlightClusters.end())
            return;
      }

      itr->fFuture.wait();

      std::lock_guard<std::mutex> lockGuardInFlightClusters(fLockWorkQueue);
      fInFlightClusters.erase(itr);
   }
}
