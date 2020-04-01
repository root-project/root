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

#include <future>
#include <iostream>
#include <mutex>
#include <utility>

ROOT::Experimental::Detail::RClusterPool::RClusterPool(RPageSource *pageSource)
   : fPageSource(pageSource)
   , fThreadIo(&RClusterPool::ExecLoadClusters, this)
{
}

ROOT::Experimental::Detail::RClusterPool::~RClusterPool()
{
   {
      std::unique_lock<std::mutex> lock(fLockWorkQueue);
      fCvHasSpace.wait(lock, [&]{ return fWorkQueue.size() < kWorkQueueLimit; });
      fWorkQueue.emplace(RWorkItemGroup());
      if (fWorkQueue.size() == 1)
         fCvHasWork.notify_one();
   }
   fThreadIo.join();
}

void ROOT::Experimental::Detail::RClusterPool::ExecLoadClusters()
{
   while (true) {
      RWorkItemGroup workItems;
      {
         std::unique_lock<std::mutex> lock(fLockWorkQueue);
         fCvHasWork.wait(lock, [&]{ return !fWorkQueue.empty(); });
         workItems = std::move(fWorkQueue.front());
         fWorkQueue.pop();
         if (fWorkQueue.empty())
            fCvHasSpace.notify_one();
      }

      if (workItems.empty())
         break;

      //workItem.fPromise.set_value(fPageSource.LoadCluster(workItem.fClusterId));
   }
}

std::shared_ptr<ROOT::Experimental::Detail::RCluster>
ROOT::Experimental::Detail::RClusterPool::GetCluster(ROOT::Experimental::DescriptorId_t clusterId)
{
   std::lock_guard<std::mutex> lockGuard(fLock);

   auto nextId = kInvalidDescriptorId;

   if (!fCurrent) {
      //fCurrent = fPageSource.LoadCluster(clusterId);
      nextId = fPageSource->GetDescriptor().FindNextClusterId(clusterId);
   }

   auto cluster = fCurrent;
   if (cluster->GetId() != clusterId) {
      cluster = std::move(fNext.get());
      if (cluster->GetId() != clusterId) {
         //cluster = std::move(fPageSource.LoadCluster(clusterId));
      }
      fCurrent = cluster;
      nextId = fPageSource->GetDescriptor().FindNextClusterId(clusterId);
   }

   if (nextId != kInvalidDescriptorId) {
      std::promise<std::unique_ptr<RCluster>> promise;
      fNext = promise.get_future();
      RWorkItem workItem;
      workItem.fPromise = std::move(promise);
      workItem.fClusterId = nextId;
      //RWorkItemGroup itemGroup{workItem};
      RWorkItemGroup itemGroup;
      itemGroup.emplace_back(std::move(workItem));

      std::unique_lock<std::mutex> lock(fLockWorkQueue);
      fCvHasSpace.wait(lock, [&]{ return fWorkQueue.size() < kWorkQueueLimit; });
      fWorkQueue.emplace(std::move(itemGroup));
      if (fWorkQueue.size() == 1)
         fCvHasWork.notify_one();
   }

   return cluster;
}

