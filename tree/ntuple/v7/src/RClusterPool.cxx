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
   if (fClusterId == other.fClusterId) {
      if (fColumns.size() == other.fColumns.size()) {
         for (auto itr1 = fColumns.begin(), itr2 = other.fColumns.begin(); itr1 != fColumns.end(); ++itr1, ++itr2) {
            if (*itr1 == *itr2)
               continue;
            return *itr1 < *itr2;
         }
         // *this == other
         return false;
      }
      return fColumns.size() < other.fColumns.size();
   }
   return fClusterId < other.fClusterId;
}

ROOT::Experimental::Detail::RClusterPool::RClusterPool(RPageSource &pageSource, unsigned int size)
   : fPageSource(pageSource)
   , fPool(size)
   , fThreadIo(&RClusterPool::ExecLoadClusters, this)
{
   R__ASSERT(size > 0);
   fWindowPre = 0;
   fWindowPost = size;
   // Large pools maintain a small look-back window together with the large look-ahead window
   while ((1u << fWindowPre) < (fWindowPost - (fWindowPre + 1))) {
      fWindowPre++;
      fWindowPost--;
   }
}

ROOT::Experimental::Detail::RClusterPool::~RClusterPool()
{
   {
      // Controlled shutdown of the I/O thread
      std::unique_lock<std::mutex> lock(fLockWorkQueue);
      fWorkQueue.emplace(RWorkItem());
      fCvHasWork.notify_one();
   }
   fThreadIo.join();
}

void ROOT::Experimental::Detail::RClusterPool::ExecLoadClusters()
{
   while (true) {
      std::vector<RWorkItem> workItems;
      {
         std::unique_lock<std::mutex> lock(fLockWorkQueue);
         fCvHasWork.wait(lock, [&]{ return !fWorkQueue.empty(); });
         while (!fWorkQueue.empty()) {
            workItems.emplace_back(std::move(fWorkQueue.front()));
            fWorkQueue.pop();
         }
      }

      for (auto &item : workItems) {
         if (item.fClusterId == kInvalidDescriptorId)
            return;

         // TODO(jblomer): the page source needs to be capable of loading multiple clusters in one go
         auto cluster = fPageSource.LoadCluster(item.fClusterId, item.fColumns);

         // Meanwhile, the user might have requested clusters outside the look-ahead window, so that we don't
         // need the cluster anymore, in which case we simply discard it right away, before moving it to the pool
         bool discard = false;
         {
            std::unique_lock<std::mutex> lock(fLockWorkQueue);
            for (auto &inFlight : fInFlightClusters) {
               if (inFlight.fClusterId != item.fClusterId)
                  continue;
               discard = inFlight.fIsExpired;
               break;
            }
         }
         if (discard)
            cluster.reset();

         item.fPromise.set_value(std::move(cluster));
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
   using ColumnSet_t = ROOT::Experimental::Detail::RPageSource::ColumnSet_t;

private:
   std::map<DescriptorId_t, ColumnSet_t> fMap;

public:
   void Insert(DescriptorId_t clusterId, const ColumnSet_t &columns)
   {
      fMap.emplace(clusterId, columns);
   }

   bool Contains(DescriptorId_t clusterId) {
      return fMap.count(clusterId) > 0;
   }

   void Erase(DescriptorId_t clusterId, const ColumnSet_t &columns)
   {
      auto itr = fMap.find(clusterId);
      if (itr == fMap.end())
         return;
      ColumnSet_t d;
      std::copy_if(itr->second.begin(), itr->second.end(), std::inserter(d, d.end()),
         [&columns] (DescriptorId_t needle) { return columns.count(needle) == 0; });
      if (d.empty()) {
         fMap.erase(itr);
      } else {
         itr->second = d;
      }
   }

   decltype(fMap)::iterator begin() { return fMap.begin(); }
   decltype(fMap)::iterator end() { return fMap.end(); }
};

} // anonymous namespace

ROOT::Experimental::Detail::RCluster *
ROOT::Experimental::Detail::RClusterPool::GetCluster(
   DescriptorId_t clusterId, const RPageSource::ColumnSet_t &columns)
{
   const auto &desc = fPageSource.GetDescriptor();

   // Determine previous cluster ids that we keep if they happen to be in the pool
   std::set<DescriptorId_t> keep;
   auto prev = clusterId;
   for (unsigned int i = 0; i < fWindowPre; ++i) {
      prev = desc.FindPrevClusterId(prev);
      if (prev == kInvalidDescriptorId)
         break;
      keep.insert(prev);
   }

   // Determine following cluster ids and the column ids that we want to make available
   RProvides provide;
   provide.Insert(clusterId, columns);
   auto next = clusterId;
   for (unsigned int i = 1; i < fWindowPost; ++i) {
      next = desc.FindNextClusterId(next);
      if (next == kInvalidDescriptorId)
         break;
      provide.Insert(next, columns);
   }

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
      std::lock_guard<std::mutex> lockGuardInFlightClusters(fLockWorkQueue);

      for (auto itr = fInFlightClusters.begin(); itr != fInFlightClusters.end(); ) {
         R__ASSERT(itr->fFuture.valid());
         itr->fIsExpired = !provide.Contains(itr->fClusterId) && (keep.count(itr->fClusterId) == 0);

         if (itr->fFuture.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            // Remove the set of columns that are already scheduled for being loaded
            provide.Erase(itr->fClusterId, itr->fColumns);
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

      // Update the work queue and the in-flight cluster list with new requests. We already hold the work queue
      // mutex
      // TODO(jblomer): we should ensure that clusterId is given first to the I/O thread.  That is usually the
      // case but it's not ensured by the code
      for (const auto &kv : provide) {
         R__ASSERT(!kv.second.empty());

         RWorkItem workItem;
         workItem.fClusterId = kv.first;
         workItem.fColumns = kv.second;

         RInFlightCluster inFlightCluster;
         inFlightCluster.fClusterId = kv.first;
         inFlightCluster.fColumns = kv.second;
         inFlightCluster.fFuture = workItem.fPromise.get_future();
         fInFlightClusters.emplace_back(std::move(inFlightCluster));

         fWorkQueue.emplace(std::move(workItem));
      }
      if (fWorkQueue.size() > 0)
         fCvHasWork.notify_one();
   } // work queue lock guard

   return WaitFor(clusterId, columns);
}


ROOT::Experimental::Detail::RCluster *
ROOT::Experimental::Detail::RClusterPool::WaitFor(
   DescriptorId_t clusterId, const RPageSource::ColumnSet_t &columns)
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
            if (itr->fClusterId == clusterId)
               break;
         }
         R__ASSERT(itr != fInFlightClusters.end());
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
