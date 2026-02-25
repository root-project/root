/// \file ROOT/RNTupleWriteOptionsDaos.hxx
/// \ingroup NTuple
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

#ifndef ROOT_RNTupleWriteOptionsDaos
#define ROOT_RNTupleWriteOptionsDaos

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
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleWriteOptionsDaos
