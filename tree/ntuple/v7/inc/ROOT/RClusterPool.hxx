/// \file ROOT/RClusterPool.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-09-02
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RClusterPool
#define ROOT7_RClusterPool

#include <ROOT/RCluster.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <memory>
#include <mutex>
#include <future>

namespace ROOT {
namespace Experimental {
namespace Detail {

class RPageSource;

class RClusterPool {
private:
   RPageSource &fPageSource;
   std::shared_ptr<RCluster> fCurrent;
   std::future<std::unique_ptr<RCluster>> fNext;
   std::mutex fLock;

public:
   explicit RClusterPool(RPageSource &pageSource) : fPageSource(pageSource) {}
   explicit RClusterPool(const RClusterPool &other) = delete;
   RClusterPool &operator =(const RClusterPool &other) = delete;

   std::shared_ptr<RCluster> GetCluster(DescriptorId_t clusterId);
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
