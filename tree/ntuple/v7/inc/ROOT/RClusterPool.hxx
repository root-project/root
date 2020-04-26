/// \file ROOT/RClusterPool.hxx
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

#ifndef ROOT7_RClusterPool
#define ROOT7_RClusterPool

#include <ROOT/RCluster.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx> // for ColumnSet_t

#include <condition_variable>
#include <memory>
#include <mutex>
#include <future>
#include <queue>
#include <thread>
#include <set>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace Detail {

class RPageSource;

// clang-format off
/**
\class ROOT::Experimental::Detail::RClusterPool
\ingroup NTuple
\brief Managed a set of clusters containing compressed and packed pages

The cluster pool steers the preloading of (partial) clusters.
*/
// clang-format on
class RClusterPool {
private:
   /// Maximum number of queued cluster requests for the I/O thread. A single request can span mutliple clusters.
   static constexpr unsigned int kWorkQueueLimit = 4;

   /// Request to load a subset of the columns of a particular cluster.
   /// Work items come in groups and are executed by the page source.
   struct RWorkItem {
      std::promise<std::unique_ptr<RCluster>> fPromise;
      DescriptorId_t fClusterId = kInvalidDescriptorId;
      RPageSource::ColumnSet_t fColumns;
      RWorkItem() = default;
   };

   /// Clusters that are currently being processed by the I/O thread.  Every in flight cluster has a corresponding
   /// work item.
   struct RInFlightCluster {
      std::future<std::unique_ptr<RCluster>> fFuture;
      DescriptorId_t fClusterId = kInvalidDescriptorId;
      RPageSource::ColumnSet_t fColumns;
      /// By the time a cluster has been loaded, it might not be necessary anymore.
      bool fIsExpired = false;
      RInFlightCluster() = default;
      bool operator== (const RInFlightCluster &other) const { return fClusterId == other.fClusterId && fColumns == other.fColumns; }
      bool operator!= (const RInFlightCluster &other) const { return !(*this == other); }
      /// First order by cluster id, then by number of columns, than by the column ids
      bool operator< (const RInFlightCluster &other) const;
   };

   RPageSource *fPageSource;
   /// The number of clusters before the currently active cluster that should stay in the pool if present
   unsigned int fWindowPre;
   /// The number of desired clusters in the pool, including the currently active cluster
   unsigned int fWindowPost;
   /// The cache of clusters around the currently active cluster
   std::vector<std::shared_ptr<RCluster>> fPool;

   std::mutex fLockInFlightClusters;
   /// The clusters that were handed off to the I/O thread
   std::vector<RInFlightCluster> fInFlightClusters;

   std::mutex fLockWorkQueue;
   std::condition_variable fCvHasWork;
   /// The communication channel to the I/O thread
   std::queue<RWorkItem> fWorkQueue;

   std::thread fThreadIo;

   /// Every cluster id has at most one corresponding RCluster pointer in the pool
   std::shared_ptr<RCluster> FindInPool(DescriptorId_t clusterId);
   /// Returns an index of an unused element in fPool; caller has to ensure that a free slot exists
   size_t FindFreeSlot();
   /// The I/O thread routine
   void ExecLoadClusters();
   /// Returns the given cluster from the pool, which needs to contain at least the columns `columns`.
   /// Executed at the end of GetCluster when all missing data pieces have been sent to the load queue.
   /// Ideally, the function returns without blocking if the cluster is already in the pool.
   std::shared_ptr<RCluster> WaitFor(DescriptorId_t clusterId, const RPageSource::ColumnSet_t &columns);

public:
   static const unsigned int kDefaultPoolSize = 4;
   RClusterPool(RPageSource *pageSource, unsigned int size);
   explicit RClusterPool(RPageSource *pageSource) : RClusterPool(pageSource, kDefaultPoolSize) {}
   explicit RClusterPool(const RClusterPool &other) = delete;
   RClusterPool &operator =(const RClusterPool &other) = delete;
   ~RClusterPool();

   unsigned int GetWindowPre() const { return fWindowPre; }
   unsigned int GetWindowPost() const { return fWindowPost; }

   /// Returns the requested cluster either from the pool or, in case of a cache miss, lets the I/O thread load
   /// the cluster in the pool and returns it.  Triggers along the way the background loading of the following
   /// fWindowPost number of clusters.  The returned cluster has at least all the pages of `columns` and possibly
   /// pages of other columns, too.
   std::shared_ptr<RCluster> GetCluster(DescriptorId_t clusterId, const RPageSource::ColumnSet_t &columns);
}; // class RClusterPool

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
