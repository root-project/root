/// \file ROOT/RSealedPage.hxx
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2026-20-11

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RSealedPage
#define ROOT_RSealedPage

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleTypes.hxx>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>

namespace ROOT {
namespace Internal {

/// A sealed page contains the bytes of a page as written to storage (packed & compressed).  It is used
/// as an input to UnsealPages() as well as to transfer pages between different storage media.
/// RSealedPage does _not_ own the buffer it is pointing to in order to not interfere with the memory management
/// of concrete page sink and page source implementations.
struct RSealedPage {
private:
   const void *fBuffer = nullptr;
   std::size_t fBufferSize = 0; ///< Size of the page payload and the trailing checksum (if available)
   std::uint32_t fNElements = 0;
   bool fHasChecksum = false; ///< If set, the last 8 bytes of the buffer are the xxhash of the rest of the buffer

public:
   /// The page checksum is a 64bit xxhash3
   static constexpr std::size_t kNBytesPageChecksum = sizeof(std::uint64_t);

   RSealedPage() = default;
   RSealedPage(const void *buffer, std::size_t bufferSize, std::uint32_t nElements, bool hasChecksum = false)
      : fBuffer(buffer), fBufferSize(bufferSize), fNElements(nElements), fHasChecksum(hasChecksum)
   {
   }
   RSealedPage(const RSealedPage &other) = default;
   RSealedPage &operator=(const RSealedPage &other) = default;
   RSealedPage(RSealedPage &&other) = default;
   RSealedPage &operator=(RSealedPage &&other) = default;

   const void *GetBuffer() const { return fBuffer; }
   void SetBuffer(const void *buffer) { fBuffer = buffer; }

   std::size_t GetDataSize() const
   {
      assert(fBufferSize >= fHasChecksum * kNBytesPageChecksum);
      return fBufferSize - fHasChecksum * kNBytesPageChecksum;
   }
   std::size_t GetBufferSize() const { return fBufferSize; }
   void SetBufferSize(std::size_t bufferSize) { fBufferSize = bufferSize; }

   std::uint32_t GetNElements() const { return fNElements; }
   void SetNElements(std::uint32_t nElements) { fNElements = nElements; }

   bool GetHasChecksum() const { return fHasChecksum; }
   void SetHasChecksum(bool hasChecksum) { fHasChecksum = hasChecksum; }

   void ChecksumIfEnabled();
   RResult<void> VerifyChecksumIfEnabled() const;
   /// Returns a failure if the sealed page has no checksum
   RResult<std::uint64_t> GetChecksum() const;
};

using SealedPageSequence_t = std::deque<RSealedPage>;
/// A range of sealed pages referring to the same column that can be used for vector commit
struct RSealedPageGroup {
   ROOT::DescriptorId_t fPhysicalColumnId;
   SealedPageSequence_t::const_iterator fFirst;
   SealedPageSequence_t::const_iterator fLast;

   RSealedPageGroup() = default;
   RSealedPageGroup(ROOT::DescriptorId_t d, const SealedPageSequence_t::const_iterator &b,
                    const SealedPageSequence_t::const_iterator &e)
      : fPhysicalColumnId(d), fFirst(b), fLast(e)
   {
   }
};

} // namespace Internal
} // namespace ROOT

#endif // RSealedPage
