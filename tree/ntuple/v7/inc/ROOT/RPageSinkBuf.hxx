/// \file ROOT/RPageSinkBuf.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Max Orok <maxwellorok@gmail.com>
/// \author Javier Lopez-Gomez <javier.lopez.gomez@cern.ch>
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
         void AllocateSealedPageBuf() { fBuf = std::unique_ptr<unsigned char[]>(new unsigned char[fPage.GetNBytes()]); }
      };
   public:
      RColumnBuf() = default;
      RColumnBuf(const RColumnBuf&) = delete;
      RColumnBuf& operator=(const RColumnBuf&) = delete;
      RColumnBuf(RColumnBuf&&) = default;
      RColumnBuf& operator=(RColumnBuf&&) = default;
      ~RColumnBuf() { DropBufferedPages(); }

      /// Returns a reference to the newly buffered page. The reference remains
      /// valid until the return value of DrainBufferedPages() is destroyed.
      /// Note that `BufferPage()` yields the ownership of `page` to RColumnBuf.
      RPageZipItem &BufferPage(
         RPageStorage::ColumnHandle_t columnHandle, const RPage &page)
      {
         if (!fCol) {
            fCol = columnHandle;
         }
         // Safety: Insertion at the end of a deque never invalidates references
         // to existing elements.
         fBufferedPages.push_back(RPageZipItem(page));
         return fBufferedPages.back();
      }
      const RPageStorage::ColumnHandle_t &GetHandle() const { return fCol; }
      bool IsEmpty() const { return fBufferedPages.empty(); }
      bool HasSealedPagesOnly() const { return fBufferedPages.size() == fSealedPages.size(); }
      const RPageStorage::SealedPageSequence_t &GetSealedPages() const { return fSealedPages; }

      using BufferedPages_t = std::tuple<std::deque<RPageZipItem>, RPageStorage::SealedPageSequence_t>;
      /// When the return value of DrainBufferedPages() is destroyed, all references
      /// returned by GetBuffer are invalidated.
      /// This function gives up on the ownership of the buffered pages.  Thus, `ReleasePage()` must be called
      /// accordingly.
      BufferedPages_t DrainBufferedPages()
      {
         BufferedPages_t drained;
         std::swap(fBufferedPages, std::get<decltype(fBufferedPages)>(drained));
         std::swap(fSealedPages, std::get<decltype(fSealedPages)>(drained));
         return drained;
      }
      void DropBufferedPages();

      // The returned reference points to a default-constructed RSealedPage. It can be used
      // to fill in data after sealing.
      RSealedPage &RegisterSealedPage()
      {
         return fSealedPages.emplace_back();
      }

   private:
      RPageStorage::ColumnHandle_t fCol;
      /// Using a deque guarantees that element iterators are never invalidated
      /// by appends to the end of the iterator by BufferPage.
      std::deque<RPageZipItem> fBufferedPages;
      /// Pages that have been already sealed by a concurrent task. A vector commit can be issued if all
      /// buffered pages have been sealed.
      /// Note that each RSealedPage refers to the same buffer as `fBufferedPages[i].fBuf` for some value of `i`, and
      /// thus owned by RPageZipItem
      RPageStorage::SealedPageSequence_t fSealedPages;
   };

private:
   /// I/O performance counters that get registered in fMetrics
   struct RCounters {
      RNTuplePlainCounter &fParallelZip;
   };
   std::unique_ptr<RCounters> fCounters;
   /// The inner sink, responsible for actually performing I/O.
   std::unique_ptr<RPageSink> fInnerSink;
   /// The buffered page sink maintains a copy of the RNTupleModel for the inner sink.
   /// For the unbuffered case, the RNTupleModel is instead managed by a RNTupleWriter.
   std::unique_ptr<RNTupleModel> fInnerModel;
   /// Vector of buffered column pages. Indexed by column id.
   std::vector<RColumnBuf> fBufferedColumns;
   DescriptorId_t fNFields = 0;
   DescriptorId_t fNColumns = 0;

   void ConnectFields(const std::vector<RFieldBase *> &fields, NTupleSize_t firstEntry);

public:
   explicit RPageSinkBuf(std::unique_ptr<RPageSink> inner);
   RPageSinkBuf(const RPageSinkBuf&) = delete;
   RPageSinkBuf& operator=(const RPageSinkBuf&) = delete;
   RPageSinkBuf(RPageSinkBuf&&) = default;
   RPageSinkBuf& operator=(RPageSinkBuf&&) = default;
   ~RPageSinkBuf() override;

   ColumnHandle_t AddColumn(DescriptorId_t fieldId, const RColumn &column) final;

   void Create(RNTupleModel &model) final;
   void UpdateSchema(const RNTupleModelChangeset &changeset, NTupleSize_t firstEntry) final;

   void CommitPage(ColumnHandle_t columnHandle, const RPage &page) final;
   void CommitSealedPage(DescriptorId_t physicalColumnId, const RSealedPage &sealedPage) final;
   void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> ranges) final;
   std::uint64_t CommitCluster(NTupleSize_t nEntries) final;
   void CommitClusterGroup() final;
   void CommitDataset() final;

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final;
   void ReleasePage(RPage &page) final;
};

} // namespace Detail
} // namespace Experimental
} // namespace ROOT

#endif
