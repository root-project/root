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
#include <ROOT/RNTupleUtil.hxx>

#include <cstddef>
#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RNTupleCollectionWriter
\ingroup NTuple
\brief A virtual ntuple used for writing untyped collections that can be used to some extent like an RNTupleWriter
*
* This class is between a field and a ntuple.  It carries the offset column for the collection and the default entry
* taken from the collection model.  It does not, however, own an ntuple model because the collection model has been
* merged into the larger ntuple model.
*/
// clang-format on
class RNTupleCollectionWriter {
   friend class RCollectionField;

private:
   std::size_t fBytesWritten = 0;
   ClusterSize_t fOffset;
   std::unique_ptr<REntry> fDefaultEntry;

public:
   explicit RNTupleCollectionWriter(std::unique_ptr<REntry> defaultEntry)
      : fOffset(0), fDefaultEntry(std::move(defaultEntry))
   {
   }
   RNTupleCollectionWriter(const RNTupleCollectionWriter &) = delete;
   RNTupleCollectionWriter &operator=(const RNTupleCollectionWriter &) = delete;
   ~RNTupleCollectionWriter() = default;

   std::size_t Fill() { return Fill(*fDefaultEntry); }
   std::size_t Fill(REntry &entry)
   {
      const std::size_t bytesWritten = entry.Append();
      fBytesWritten += bytesWritten;
      fOffset++;
      return bytesWritten;
   }

   ClusterSize_t *GetOffsetPtr() { return &fOffset; }
}; // class RNTupleCollectionWriter

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleCollectionWriter
