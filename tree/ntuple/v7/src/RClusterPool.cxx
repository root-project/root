/// \file RClusterPool.cxx
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

#include <ROOT/RClusterPool.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RPageStorage.hxx>

#include <future>
#include <iostream>
#include <mutex>
#include <utility>

std::shared_ptr<ROOT::Experimental::Detail::RCluster>
ROOT::Experimental::Detail::RClusterPool::GetCluster(ROOT::Experimental::DescriptorId_t clusterId)
{
	std::lock_guard<std::mutex> lockGuard(fLock);

	if (!fCurrent) {
		fCurrent = fPageSource.LoadCluster(clusterId);
		auto nextId = fPageSource.GetDescriptor().FindNextClusterId(clusterId);
		std::cout << "CLUSTER ASYNC, next id is " << nextId << std::endl;
		if (nextId != kInvalidDescriptorId) {
			// What happens if thread still runs on delete of the cluster pool
			fNext = std::async(std::launch::async, &RPageSource::LoadCluster, &fPageSource, nextId);
		}
		return fCurrent;
	}

   auto cluster = fCurrent;
	if (cluster->GetId() == clusterId)
		return cluster;

	cluster = std::move(fNext.get());
	if (cluster->GetId() != clusterId) {
		cluster = std::move(fPageSource.LoadCluster(clusterId));
	}
	fCurrent = cluster;

	auto nextId = fPageSource.GetDescriptor().FindNextClusterId(clusterId);
	std::cout << "CLUSTER ASYNC, next id is " << nextId << std::endl;
	if (nextId != kInvalidDescriptorId) {
		// What happens if thread still runs on delete of the cluster pool
		fNext = std::async(std::launch::async, &RPageSource::LoadCluster, &fPageSource, nextId);
	}

	return cluster;
}
