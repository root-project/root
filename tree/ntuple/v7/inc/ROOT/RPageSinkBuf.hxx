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
#include <functional>
#include <iterator>
#include <memory>
#include <tuple>

namespace ROOT {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageSinkBuf
\ingroup NTuple
\brief Wrapper sink that coalesces cluster column page writes
*/
// clang-format on
class RPageSinkBuf : public RPageSink {
private:
   /// A buffered column. The column is not responsible for RPage memory management (i.e. ReservePage),
   /// which is handled by the enclosing RPageSinkBuf.
   class RColumnBuf {
   public:
      struct RPageZipItem {
         ROOT::Internal::RPage fPage;
         // Compression scratch buffer for fSealedPage.
         std::unique_ptr<unsigned char[]> fBuf;
         RPageStorage::RSealedPage *fSealedPage = nullptr;
         bool IsSealed() const { return fSealedPage != nullptr; }
      };
   public:
      RColumnBuf() = default;
      RColumnBuf(const RColumnBuf&) = delete;
      RColumnBuf& operator=(const RColumnBuf&) = delete;
      RColumnBuf(RColumnBuf&&) = default;
      RColumnBuf& operator=(RColumnBuf&&) = default;
      ~RColumnBuf() { DropBufferedPages(); }

      /// Returns a reference to the newly buffered page. The reference remains
      /// valid until DropBufferedPages().
      RPageZipItem &BufferPage(RPageStorage::ColumnHandle_t columnHandle)
      {
         if (!fCol) {
            fCol = columnHandle;
         }
         // Safety: Insertion at the end of a deque never invalidates references
         // to existing elements.
         return fBufferedPages.emplace_back();
      }
      const RPageStorage::ColumnHandle_t &GetHandle() const { return fCol; }
      bool IsEmpty() const { return fBufferedPages.empty(); }
      bool HasSealedPagesOnly() const { return fBufferedPages.size() == fSealedPages.size(); }
      const RPageStorage::SealedPageSequence_t &GetSealedPages() const { return fSealedPages; }

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
      Detail::RNTuplePlainCounter &fParallelZip;
      Detail::RNTupleAtomicCounter &fTimeWallZip;
      Detail::RNTuplePlainCounter &fTimeWallCriticalSection;
      Detail::RNTupleTickCounter<Detail::RNTupleAtomicCounter> &fTimeCpuZip;
      Detail::RNTupleTickCounter<Detail::RNTuplePlainCounter> &fTimeCpuCriticalSection;
   };
   std::unique_ptr<RCounters> fCounters;
   /// The inner sink, responsible for actually performing I/O.
   std::unique_ptr<RPageSink> fInnerSink;
   /// The buffered page sink maintains a copy of the RNTupleModel for the inner sink.
   /// For the unbuffered case, the RNTupleModel is instead managed by a RNTupleWriter.
   std::unique_ptr<RNTupleModel> fInnerModel;
   /// Vector of buffered column pages. Indexed by column id.
   std::vector<RColumnBuf> fBufferedColumns;
   /// Columns committed as suppressed are stored and passed to the inner sink at cluster commit
   std::vector<ColumnHandle_t> fSuppressedColumns;
   ROOT::DescriptorId_t fNFields = 0;
   ROOT::DescriptorId_t fNColumns = 0;

   void ConnectFields(const std::vector<RFieldBase *> &fields, ROOT::NTupleSize_t firstEntry);
   void FlushClusterImpl(std::function<void(void)> FlushClusterFn);

public:
   explicit RPageSinkBuf(std::unique_ptr<RPageSink> inner);
   RPageSinkBuf(const RPageSinkBuf&) = delete;
   RPageSinkBuf& operator=(const RPageSinkBuf&) = delete;
   RPageSinkBuf(RPageSinkBuf&&) = default;
   RPageSinkBuf& operator=(RPageSinkBuf&&) = default;
   ~RPageSinkBuf() override;

   ColumnHandle_t AddColumn(ROOT::DescriptorId_t fieldId, RColumn &column) final;

   const RNTupleDescriptor &GetDescriptor() const final;

   ROOT::NTupleSize_t GetNEntries() const final { return fInnerSink->GetNEntries(); }

   void InitImpl(RNTupleModel &model) final;
   void UpdateSchema(const RNTupleModelChangeset &changeset, ROOT::NTupleSize_t firstEntry) final;
   void UpdateExtraTypeInfo(const RExtraTypeInfoDescriptor &extraTypeInfo) final;

   void CommitSuppressedColumn(ColumnHandle_t columnHandle) final;
   void CommitPage(ColumnHandle_t columnHandle, const ROOT::Internal::RPage &page) final;
   void CommitSealedPage(ROOT::DescriptorId_t physicalColumnId, const RSealedPage &sealedPage) final;
   void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> ranges) final;
   std::uint64_t CommitCluster(ROOT::NTupleSize_t nNewEntries) final;
   RStagedCluster StageCluster(ROOT::NTupleSize_t nNewEntries) final;
   void CommitStagedClusters(std::span<RStagedCluster> clusters) final;
   void CommitClusterGroup() final;
   void CommitDatasetImpl() final;

   ROOT::Internal::RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final;
}; // RPageSinkBuf

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
