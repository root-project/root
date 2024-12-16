/// \file RNTupleWriteOptions.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

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

std::unique_ptr<ROOT::Experimental::RNTupleWriteOptions> ROOT::Experimental::RNTupleWriteOptions::Clone() const
{
   return std::make_unique<RNTupleWriteOptions>(*this);
}

void ROOT::Experimental::RNTupleWriteOptions::SetApproxZippedClusterSize(std::size_t val)
{
   EnsureValidTunables(val, fMaxUnzippedClusterSize, fInitialUnzippedPageSize, fMaxUnzippedPageSize);
   fApproxZippedClusterSize = val;
}

void ROOT::Experimental::RNTupleWriteOptions::SetMaxUnzippedClusterSize(std::size_t val)
{
   EnsureValidTunables(fApproxZippedClusterSize, val, fInitialUnzippedPageSize, fMaxUnzippedPageSize);
   fMaxUnzippedClusterSize = val;
}

void ROOT::Experimental::RNTupleWriteOptions::SetInitialUnzippedPageSize(std::size_t val)
{
   EnsureValidTunables(fApproxZippedClusterSize, fMaxUnzippedClusterSize, val, fMaxUnzippedPageSize);
   fInitialUnzippedPageSize = val;
}

void ROOT::Experimental::RNTupleWriteOptions::SetMaxUnzippedPageSize(std::size_t val)
{
   EnsureValidTunables(fApproxZippedClusterSize, fMaxUnzippedClusterSize, fInitialUnzippedPageSize, val);
   fMaxUnzippedPageSize = val;
}

std::size_t ROOT::Experimental::RNTupleWriteOptions::GetPageBufferBudget() const
{
   if (fPageBufferBudget != 0)
      return fPageBufferBudget;

   return GetApproxZippedClusterSize() + (GetCompression() != 0) * GetApproxZippedClusterSize();
}
