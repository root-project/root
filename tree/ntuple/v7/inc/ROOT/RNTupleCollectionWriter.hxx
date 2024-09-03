/// \file ROOT/RNTupleCollectionWriter.hxx
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

#ifndef ROOT7_RNTupleCollectionWriter
#define ROOT7_RNTupleCollectionWriter

#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleCollectionWriter
\ingroup NTuple
\brief A special type only produced by the collection field and used for writing untyped collections

This class is tightly coupled to the RCollectionField. It fills the sub fields of the collection fields one-by-one.
An instance can only be used with the exact RCollectionField that created it. Upon creation, the entry values need
to be bound to memory locations.
*/
// clang-format on
class RNTupleCollectionWriter {
   friend class RCollectionField;

private:
   enum class EEnvironmentState {
      kBorn,            // created from RCollectionField
      kConnectedToSink, // the collection field that created the writer is connected to a sink
      kOrphaned,        // the collection field that created the writer is destructed
   };

   // fNBytesWritten and fNElements are reset to zero by RCollectionField::Append and thus need to be mutable
   mutable std::size_t fNBytesWritten = 0;
   mutable ClusterSize_t fNElements;
   /// The collection writer depends on the RCollectionField from which it was created. Filling collection elements
   /// only works when the collection field is connected to a page sink, and as long as the collection field is alive.
   /// This member is set by the collection field, who keeps track of all the writers it creates.
   std::atomic<EEnvironmentState> fEnvironmentState = EEnvironmentState::kBorn;
   /// The entry is constructed by the RCollectionField on construction of the collection writer.
   /// Its values have their fields point to the subfields of the collection field.
   /// The entry is bare and memory locations need to be bound to it before Fill() can be used.
   REntry fEntry;

   // Constructed by RCollectionField::ConstructValue
   explicit RNTupleCollectionWriter(REntry &&entry) : fNElements(0), fEntry(std::move(entry)) {}

   // Called by the RCollectionField after append
   void Reset() const
   {
      fNBytesWritten = 0;
      fNElements = 0;
   }

public:
   RNTupleCollectionWriter(const RNTupleCollectionWriter &) = delete;
   RNTupleCollectionWriter &operator=(const RNTupleCollectionWriter &) = delete;
   RNTupleCollectionWriter(RNTupleCollectionWriter &&) = delete;
   RNTupleCollectionWriter &operator=(RNTupleCollectionWriter &&) = delete;
   ~RNTupleCollectionWriter() = default;

   std::size_t Fill()
   {
      if (fEnvironmentState != EEnvironmentState::kConnectedToSink) {
         throw RException(R__FAIL("invalid attempt to fill an untyped collection element without a valid field"));
      }
      const std::size_t nBytesWritten = fEntry.Append();
      fNBytesWritten += nBytesWritten;
      fNElements++;
      return nBytesWritten;
   }

   const REntry &GetEntry() const { return fEntry; }
   REntry &GetEntry() { return fEntry; }
}; // class RNTupleCollectionWriter

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleCollectionWriter
