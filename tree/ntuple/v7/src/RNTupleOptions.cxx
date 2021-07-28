/// \file RNTupleOptions.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2021-07-28
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleOptions.hxx>

#include <utility>

namespace {

void EnsureValidTunables(std::size_t zippedClusterSize, std::size_t unzippedClusterSize, std::size_t unzippedPageSize)
{
   using RException = ROOT::Experimental::RException;
   if (zippedClusterSize == 0) {
      throw RException(R__FAIL("invalid target cluster size: 0"));
   }
   if (zippedClusterSize > unzippedClusterSize) {
      throw RException(R__FAIL("compressed target cluster size must not be larger than "
                               "maximum uncompressed cluster size"));
   }
   if (unzippedPageSize > unzippedClusterSize) {
      throw RException(R__FAIL("compressed target page size must not be larger than "
                               "maximum uncompressed cluster size"));
   }
   if (unzippedPageSize == 0) {
      throw RException(R__FAIL("invalid target page size: 0"));
   }
}

} // anonymous namespace

std::unique_ptr<ROOT::Experimental::RNTupleWriteOptions>
ROOT::Experimental::RNTupleWriteOptions::Clone() const
{
   return std::make_unique<RNTupleWriteOptions>(*this);
}

void ROOT::Experimental::RNTupleWriteOptions::SetApproxZippedClusterSize(std::size_t val)
{
   EnsureValidTunables(val, fMaxUnzippedClusterSize, fApproxUnzippedPageSize);
   fApproxZippedClusterSize = val;
}

void ROOT::Experimental::RNTupleWriteOptions::SetMaxUnzippedClusterSize(std::size_t val)
{
   EnsureValidTunables(fApproxZippedClusterSize, val, fApproxUnzippedPageSize);
   fMaxUnzippedClusterSize = val;
}

void ROOT::Experimental::RNTupleWriteOptions::SetApproxUnzippedPageSize(std::size_t val)
{
   EnsureValidTunables(fApproxZippedClusterSize, fMaxUnzippedClusterSize, val);
   fApproxUnzippedPageSize = val;
}
