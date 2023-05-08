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

bool ROOT::Experimental::Detail::RClusterPool::RInFlightCluster::operator <(const RInFlightCluster &other) const
{
   if (fClusterKey.fClusterId == other.fClusterKey.fClusterId) {
      if (fClusterKey.fColumnSet.size() == other.fClusterKey.fColumnSet.size()) {
         for (auto itr1 = fClusterKey.fColumnSet.begin(), itr2 = other.fClusterKey.fColumnSet.begin();
              itr1 != fClusterKey.fColumnSet.end(); ++itr1, ++itr2)
         {
            if (*itr1 == *itr2)
               continue;
            return *itr1 < *itr2;
         }
         // *this == other
         return false;
      }
      return fClusterKey.fColumnSet.size() < other.fClusterKey.fColumnSet.size();
   }
   return fClusterKey.fClusterId < other.fClusterKey.fClusterId;
}

ROOT::Experimental::Detail::RClusterPool::RClusterPool(RPageSource &pageSource, unsigned int clusterBunchSize)
   : fPageSource(pageSource)
   , fClusterBunchSize(clusterBunchSize)
   , fPool(2 * clusterBunchSize)
   , fThreadIo(&RClusterPool::ExecReadClusters, this)
   , fThreadUnzip(&RClusterPool::ExecUnzipClusters, this)
{
   R__ASSERT(clusterBunchSize > 0);
}

ROOT::Experimental::Detail::RClusterPool::~RClusterPool()
{
   {
      // Controlled shutdown of the I/O thread
      std::unique_lock<std::mutex> lock(fLockWorkQueue);
      fReadQueue.emplace(RReadItem());
      fCvHasReadWork.notify_one();
   }
   fThreadIo.join();

   {
      // Controlled shutdown of the unzip thread
      std::unique_lock<std::mutex> lock(fLockUnzipQueue);
      fUnzipQueue.emplace(RUnzipItem());
      fCvHasUnzipWork.notify_one();
   }
   fThreadUnzip.join();
}

void ROOT::Experimental::Detail::RClusterPool::ExecUnzipClusters()
{
   while (true) {
      std::vector<RUnzipItem> unzipItems;
      {
         std::unique_lock<std::mutex> lock(fLockUnzipQueue);
         fCvHasUnzipWork.wait(lock, [&]{ return !fUnzipQueue.empty(); });
         while (!fUnzipQueue.empty()) {
            unzipItems.emplace_back(std::move(fUnzipQueue.front()));
            fUnzipQueue.pop();
         }
      }

      for (auto &item : unzipItems) {
         if (!item.fCluster)
            return;

         fPageSource.UnzipCluster(item.fCluster.get());

         // Afterwards the GetCluster() method in the main thread can pick-up the cluster
         item.fPromise.set_value(std::move(item.fCluster));
      }
   } // while (true)
}

void ROOT::Experimental::Detail::RClusterPool::ExecReadClusters()
{
   while (true) {
      std::vector<RReadItem> readItems;
      std::vector<RCluster::RKey> clusterKeys;
      std::int64_t bunchId = -1;
      {
         std::unique_lock<std::mutex> lock(fLockWorkQueue);
         fCvHasReadWork.wait(lock, [&]{ return !fReadQueue.empty(); });
         while (!fReadQueue.empty()) {
            if (fReadQueue.front().fClusterKey.fClusterId == kInvalidDescriptorId) {
               fReadQueue.pop();
               return;
            }

            if ((bunchId >= 0) && (fReadQueue.front().fBunchId != bunchId))
               break;
            readItems.emplace_back(std::move(fReadQueue.front()));
            fReadQueue.pop();
            bunchId = readItems.back().fBunchId;
            clusterKeys.emplace_back(readItems.back().fClusterKey);
         }
      }

      auto clusters = fPageSource.LoadClusters(clusterKeys);

      for (std::size_t i = 0; i < clusters.size(); ++i) {
         // Meanwhile, the user might have requested clusters outside the look-ahead window, so that we don't
         // need the cluster anymore, in which case we simply discard it right away, before moving it to the pool
         bool discard = false;
         {
            std::unique_lock<std::mutex> lock(fLockWorkQueue);
            for (auto &inFlight : fInFlightClusters) {
               if (inFlight.fClusterKey.fClusterId != clusters[i]->GetId())
                  continue;
               discard = inFlight.fIsExpired;
               break;
            }
         }
         if (discard) {
            clusters[i].reset();
            readItems[i].fPromise.set_value(std::move(clusters[i]));
         } else {
            // Hand-over the loaded cluster pages to the unzip thread
            std::unique_lock<std::mutex> lock(fLockUnzipQueue);
            fUnzipQueue.emplace(RUnzipItem{std::move(clusters[i]), std::move(readItems[i].fPromise)});
            fCvHasUnzipWork.notify_one();
         }
      }
   } // while (true)
}

ROOT::Experimental::Detail::RCluster *
ROOT::Experimental::Detail::RClusterPool::FindInPool(DescriptorId_t clusterId) const
{
   for (const auto &cptr : fPool) {
      if (cptr && (cptr->GetId() == clusterId))
         return cptr.get();
   }
   return nullptr;
}

size_t ROOT::Experimental::Detail::RClusterPool::FindFreeSlot() const
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
   using ColumnSet_t = ROOT::Experimental::Detail::RCluster::ColumnSet_t;

public:
   struct RInfo {
      std::int64_t fBunchId = -1;
      std::int64_t fFlags = 0;
      ColumnSet_t fColumnSet;
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

   void Erase(DescriptorId_t clusterId, const ColumnSet_t &columns)
   {
      auto itr = fMap.find(clusterId);
      if (itr == fMap.end())
         return;
      ColumnSet_t d;
      std::copy_if(itr->second.fColumnSet.begin(), itr->second.fColumnSet.end(), std::inserter(d, d.end()),
         [&columns] (DescriptorId_t needle) { return columns.count(needle) == 0; });
      if (d.empty()) {
         fMap.erase(itr);
      } else {
         itr->second.fColumnSet = d;
      }
   }

   decltype(fMap)::iterator begin() { return fMap.begin(); }
   decltype(fMap)::iterator end() { return fMap.end(); }
};

} // anonymous namespace

ROOT::Experimental::Detail::RCluster *
ROOT::Experimental::Detail::RClusterPool::GetCluster(
   DescriptorId_t clusterId, const RCluster::ColumnSet_t &columns)
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
      provideInfo.fColumnSet = columns;
      provideInfo.fBunchId = fBunchId;
      provideInfo.fFlags = RProvides::kFlagRequired;
      for (DescriptorId_t i = 0, next = clusterId; i < 2 * fClusterBunchSize; ++i) {
         if (i == fClusterBunchSize)
            provideInfo.fBunchId = ++fBunchId;

         auto cid = next;
         next = descriptorGuard->FindNextClusterId(cid);
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
         itr->fIsExpired =
            !provide.Contains(itr->fClusterKey.fClusterId) && (keep.count(itr->fClusterKey.fClusterId) == 0);

         if (itr->fFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            // Remove the set of columns that are already scheduled for being loaded
            provide.Erase(itr->fClusterKey.fClusterId, itr->fClusterKey.fColumnSet);
            ++itr;
            continue;
         }

         auto cptr = itr->fFuture.get();
         // If cptr is nullptr, the cluster expired previously and was released by the I/O thread
         if (!cptr || itr->fIsExpired) {
            cptr.reset();
            itr = fInFlightClusters.erase(itr);
            continue;
         }

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
         provide.Erase(cptr->GetId(), cptr->GetAvailColumns());
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
            R__ASSERT(!kv.second.fColumnSet.empty());

            RReadItem readItem;
            readItem.fClusterKey.fClusterId = kv.first;
            readItem.fBunchId = kv.second.fBunchId;
            readItem.fClusterKey.fColumnSet = kv.second.fColumnSet;

            RInFlightCluster inFlightCluster;
            inFlightCluster.fClusterKey.fClusterId = kv.first;
            inFlightCluster.fClusterKey.fColumnSet = kv.second.fColumnSet;
            inFlightCluster.fFuture = readItem.fPromise.get_future();
            fInFlightClusters.emplace_back(std::move(inFlightCluster));

            fReadQueue.emplace(std::move(readItem));
         }
         if (fReadQueue.size() > 0)
            fCvHasReadWork.notify_one();
      }
   } // work queue lock guard

   return WaitFor(clusterId, columns);
}


ROOT::Experimental::Detail::RCluster *
ROOT::Experimental::Detail::RClusterPool::WaitFor(
   DescriptorId_t clusterId, const RCluster::ColumnSet_t &columns)
{
   while (true) {
      // Fast exit: the cluster happens to be already present in the cache pool
      auto result = FindInPool(clusterId);
      if (result) {
         bool hasMissingColumn = false;
         for (auto cid : columns) {
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


void ROOT::Experimental::Detail::RClusterPool::WaitForInFlightClusters()
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
