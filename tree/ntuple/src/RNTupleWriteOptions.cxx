/// \file RNTupleWriteOptions.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-22

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>

#include <utility>

namespace {

void EnsureValidTunables(std::size_t zippedClusterSize, std::size_t unzippedClusterSize,
                         std::size_t initialUnzippedPageSize, std::size_t maxUnzippedPageSize)
{
   if (zippedClusterSize == 0) {
      throw ROOT::RException(R__FAIL("invalid target cluster size: 0"));
   }
   if (initialUnzippedPageSize == 0) {
      throw ROOT::RException(R__FAIL("invalid initial page size: 0"));
   }
   if (maxUnzippedPageSize == 0) {
      throw ROOT::RException(R__FAIL("invalid maximum page size: 0"));
   }
   if (zippedClusterSize > unzippedClusterSize) {
      throw ROOT::RException(R__FAIL("compressed target cluster size must not be larger than "
                                     "maximum uncompressed cluster size"));
   }
   if (initialUnzippedPageSize > maxUnzippedPageSize) {
      throw ROOT::RException(R__FAIL("initial page size must not be larger than maximum page size"));
   }
   if (maxUnzippedPageSize > unzippedClusterSize) {
      throw ROOT::RException(R__FAIL("maximum page size must not be larger than "
                                     "maximum uncompressed cluster size"));
   }
}

} // anonymous namespace

std::unique_ptr<ROOT::RNTupleWriteOptions> ROOT::RNTupleWriteOptions::Clone() const
{
   return std::make_unique<RNTupleWriteOptions>(*this);
}

void ROOT::RNTupleWriteOptions::SetApproxZippedClusterSize(std::size_t val)
{
   EnsureValidTunables(val, fMaxUnzippedClusterSize, fInitialUnzippedPageSize, fMaxUnzippedPageSize);
   fApproxZippedClusterSize = val;
}

void ROOT::RNTupleWriteOptions::SetMaxUnzippedClusterSize(std::size_t val)
{
   EnsureValidTunables(fApproxZippedClusterSize, val, fInitialUnzippedPageSize, fMaxUnzippedPageSize);
   fMaxUnzippedClusterSize = val;
}

void ROOT::RNTupleWriteOptions::SetInitialUnzippedPageSize(std::size_t val)
{
   EnsureValidTunables(fApproxZippedClusterSize, fMaxUnzippedClusterSize, val, fMaxUnzippedPageSize);
   fInitialUnzippedPageSize = val;
}

void ROOT::RNTupleWriteOptions::SetMaxUnzippedPageSize(std::size_t val)
{
   EnsureValidTunables(fApproxZippedClusterSize, fMaxUnzippedClusterSize, fInitialUnzippedPageSize, val);
   fMaxUnzippedPageSize = val;
}

void ROOT::RNTupleWriteOptions::SetEnableSamePageMerging(bool val)
{
   if (val && !fEnablePageChecksums) {
      throw RException(R__FAIL("same page merging requires page checksums, which were previously disabled"));
   }
   fEnableSamePageMerging = val;
}

std::size_t ROOT::RNTupleWriteOptions::GetPageBufferBudget() const
{
   if (fPageBufferBudget != 0)
      return fPageBufferBudget;

   return GetApproxZippedClusterSize() + (GetCompression() != 0) * GetApproxZippedClusterSize();
}
