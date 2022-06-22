/// \file ROOT/RPageSinkBuf.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Max Orok <maxwellorok@gmail.com>
/// \date 2021-03-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageSinkBuf
#define ROOT7_RPageSinkBuf

#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RPageStorage.hxx>

#include <deque>
#include <iterator>
#include <memory>
#include <tuple>

namespace ROOT {
namespace Experimental {
namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkBuf
\ingroup NTuple
\brief Wrapper sink that coalesces cluster column page writes
*
* TODO(jblomer): The interplay of derived class and RPageSink is not yet optimally designed for page storage wrapper
* classes like this one. Header and footer serialization, e.g., are done twice.  To be revised.
*/
// clang-format on
class RPageSinkBuf : public RPageSink {
private:
   /// A buffered column. The column is not responsible for RPage memory management (i.e.
   /// ReservePage/ReleasePage), which is handled by the enclosing RPageSinkBuf.
   class RColumnBuf {
   public:
      struct RPageZipItem {
         RPage fPage;
         // Compression scratch buffer for fSealedPage.
         std::unique_ptr<unsigned char[]> fBuf;
         RPageStorage::RSealedPage *fSealedPage = nullptr;
         explicit RPageZipItem(RPage page)
            : fPage(page), fBuf(nullptr) {}
         bool IsSealed() const { return fSealedPage != nullptr; }
         void AllocateSealedPageBuf() {
            fBuf = std::make_unique<unsigned char[]>(fPage.GetNBytes());
         }
      };
   public:
      RColumnBuf() = default;
      RColumnBuf(const RColumnBuf&) = delete;
      RColumnBuf& operator=(const RColumnBuf&) = delete;
      RColumnBuf(RColumnBuf&&) = default;
      RColumnBuf& operator=(RColumnBuf&&) = default;
      ~RColumnBuf() = default;

      using iterator = std::deque<RPageZipItem>::iterator;
      /// Returns an iterator to the newly buffered page. The iterator remains
      /// valid until the return value of DrainBufferedPages() is destroyed.
      iterator BufferPage(
         RPageStorage::ColumnHandle_t columnHandle, const RPage &page)
      {
         if (!fCol) {
            fCol = columnHandle;
         }
         // Safety: Insertion at the end of a deque never invalidates existing
         // iterators.
         fBufferedPages.push_back(RPageZipItem(page));
         return std::prev(fBufferedPages.end());
      }
      const RPageStorage::ColumnHandle_t &GetHandle() const { return fCol; }
      bool HasSealedPagesOnly() const { return fBufferedPages.size() == fSealedPages.size(); }
      const RPageStorage::SealedPageSequence_t &GetSealedPages() const { return fSealedPages; }

      using BufferedPages_t = std::tuple<std::deque<RPageZipItem>, RPageStorage::SealedPageSequence_t>;
      // When the return value of DrainBufferedPages() is destroyed, all iterators
      // returned by GetBuffer are invalidated.
      BufferedPages_t DrainBufferedPages()
      {
         BufferedPages_t drained;
         std::swap(fBufferedPages, std::get<decltype(fBufferedPages)>(drained));
         std::swap(fSealedPages, std::get<decltype(fSealedPages)>(drained));
         return drained;
      }

      // The returned iterator points to a default-constructed RSealedPage. This iterator can be used
      // to fill in data after sealing.
      RPageStorage::SealedPageSequence_t::iterator RegisterSealedPage()
      {
         return fSealedPages.emplace(std::end(fSealedPages));
      }

   private:
      RPageStorage::ColumnHandle_t fCol;
      // Using a deque guarantees that element iterators are never invalidated
      // by appends to the end of the iterator by BufferPage.
      std::deque<RPageZipItem> fBufferedPages;
      // Pages that have been already sealed by a concurrent task. A vector commit can be issued if all
      // buffered pages have been sealed.
      RPageStorage::SealedPageSequence_t fSealedPages;
   };

private:
   /// I/O performance counters that get registered in fMetrics
   struct RCounters {
      RNTuplePlainCounter &fParallelZip;
   };
   std::unique_ptr<RCounters> fCounters;
   RNTupleMetrics fMetrics;
   /// The inner sink, responsible for actually performing I/O.
   std::unique_ptr<RPageSink> fInnerSink;
   /// The buffered page sink maintains a copy of the RNTupleModel for the inner sink.
   /// For the unbuffered case, the RNTupleModel is instead managed by a RNTupleWriter.
   std::unique_ptr<RNTupleModel> fInnerModel;
   /// Vector of buffered column pages. Indexed by column id.
   std::vector<RColumnBuf> fBufferedColumns;

protected:
   void CreateImpl(const RNTupleModel &model, unsigned char *serializedHeader, std::uint32_t length) final;
   RNTupleLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) final;
   RNTupleLocator CommitSealedPageImpl(DescriptorId_t columnId, const RSealedPage &sealedPage) final;
   std::uint64_t CommitClusterImpl(NTupleSize_t nEntries) final;
   RNTupleLocator CommitClusterGroupImpl(unsigned char *serializedPageList, std::uint32_t length) final;
   void CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length) final;

public:
   explicit RPageSinkBuf(std::unique_ptr<RPageSink> inner);
   RPageSinkBuf(const RPageSinkBuf&) = delete;
   RPageSinkBuf& operator=(const RPageSinkBuf&) = delete;
   RPageSinkBuf(RPageSinkBuf&&) = default;
   RPageSinkBuf& operator=(RPageSinkBuf&&) = default;
   virtual ~RPageSinkBuf() = default;

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final;
   void ReleasePage(RPage &page) final;

   RNTupleMetrics &GetMetrics() final { return fMetrics; }
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
