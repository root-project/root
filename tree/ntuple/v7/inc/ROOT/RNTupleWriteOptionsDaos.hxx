/// \file ROOT/RNTupleWriteOptionsDaos.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-22
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleWriteOptionsDaos
#define ROOT7_RNTupleWriteOptionsDaos

#include <ROOT/RNTupleWriteOptions.hxx>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleWriteOptionsDaos
\ingroup NTuple
\brief DAOS-specific user-tunable settings for storing ntuples
*/
// clang-format on
class RNTupleWriteOptionsDaos : public ROOT::RNTupleWriteOptions {
   std::string fObjectClass{"SX"};
   /// The maximum cage size is set to the equivalent of 16 uncompressed pages - 16MiB by default.
   /// A `fMaxCageSize` of 0 disables the caging mechanism.
   uint32_t fMaxCageSize = 16 * RNTupleWriteOptions::fMaxUnzippedPageSize;

public:
   ~RNTupleWriteOptionsDaos() override = default;
   std::unique_ptr<RNTupleWriteOptions> Clone() const override
   {
      return std::make_unique<RNTupleWriteOptionsDaos>(*this);
   }

   const std::string &GetObjectClass() const { return fObjectClass; }
   /// Set the object class used to generate OIDs that relate to user data. Any
   /// `OC_xxx` constant defined in `daos_obj_class.h` may be used here without
   /// the OC_ prefix.
   void SetObjectClass(const std::string &val) { fObjectClass = val; }

   uint32_t GetMaxCageSize() const { return fMaxCageSize; }
   /// Set the upper bound for page concatenation into cages, in bytes. It is assumed
   /// that cage size will be no smaller than the approximate uncompressed page size.
   /// To disable page concatenation, set this value to 0.
   void SetMaxCageSize(uint32_t cageSz) { fMaxCageSize = cageSz; }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleWriteOptionsDaos
