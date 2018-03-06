#ifndef ROOT_TARROWTDS
#define ROOT_TARROWTDS

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <arrow/table.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <memory>

#include "ROOT/TDataFrame.hxx"
#include "ROOT/TDataSource.hxx"

namespace ROOT {
namespace Experimental {
namespace TDF {

class TArrowDS final : public TDataSource {
private:
   // Per slot visitor of an Array.
   class ArrayPtrVisitor : public ::arrow::ArrayVisitor {
   private:
      /// The pointer to update.
      void **fResult;
      bool fCachedBool;          // Booleans need to be unpacked, so we use a cached entry.
      std::string fCachedString; //
      /// The entry in the array which should be looked up.
      ULong64_t fCurrentEntry;

   public:
      ArrayPtrVisitor(void **result) : fResult{result}, fCurrentEntry{0} {}

      void SetEntry(ULong64_t entry) { fCurrentEntry = entry; }

      /// Check if we are asking the same entry as before.
      virtual arrow::Status Visit(arrow::Int32Array const &array) final
      {
         *fResult = (void *)(array.raw_values() + fCurrentEntry);
         return arrow::Status::OK();
      }

      virtual arrow::Status Visit(arrow::Int64Array const &array) final
      {
         *fResult = (void *)(array.raw_values() + fCurrentEntry);
         return arrow::Status::OK();
      }

      /// Check if we are asking the same entry as before.
      virtual arrow::Status Visit(arrow::UInt32Array const &array) final
      {
         *fResult = (void *)(array.raw_values() + fCurrentEntry);
         return arrow::Status::OK();
      }

      virtual arrow::Status Visit(arrow::UInt64Array const &array) final
      {
         *fResult = (void *)(array.raw_values() + fCurrentEntry);
         return arrow::Status::OK();
      }

      virtual arrow::Status Visit(arrow::FloatArray const &array) final
      {
         *fResult = (void *)(array.raw_values() + fCurrentEntry);
         return arrow::Status::OK();
      }

      virtual arrow::Status Visit(arrow::DoubleArray const &array) final
      {
         *fResult = (void *)(array.raw_values() + fCurrentEntry);
         return arrow::Status::OK();
      }

      virtual arrow::Status Visit(arrow::BooleanArray const &array) final
      {
         fCachedBool = array.Value(fCurrentEntry);
         *fResult = reinterpret_cast<void *>(&fCachedBool);
         return arrow::Status::OK();
      }

      virtual arrow::Status Visit(arrow::StringArray const &array) final
      {
         fCachedString = array.GetString(fCurrentEntry);
         *fResult = reinterpret_cast<void *>(&fCachedString);
         return arrow::Status::OK();
      }

      using ::arrow::ArrayVisitor::Visit;
   };

   /// Helper class which keeps track for each slot where to get the entry.
   class ValueGetter {
   private:
      std::vector<void *> fValuesPtrPerSlot;
      std::vector<ULong64_t> fLastEntryPerSlot;
      std::vector<ULong64_t> fLastChunkPerSlot;
      std::vector<ULong64_t> fFirstEntryPerChunk;
      std::vector<ArrayPtrVisitor> fArrayVisitorPerSlot;
      /// Since data can be chunked in different arrays we need to construct an
      /// index which contains the first element of each chunk, so that we can
      /// quickly move to the correct chunk.
      std::vector<ULong64_t> fChunkIndex;
      arrow::ArrayVector fChunks;

   public:
      ValueGetter(size_t slots, arrow::ArrayVector chunks)
         : fValuesPtrPerSlot(slots, nullptr), fLastEntryPerSlot(slots, 0), fLastChunkPerSlot(slots, 0), fChunks{chunks}
      {
         fChunkIndex.reserve(fChunks.size());
         size_t next = 0;
         for (auto &chunk : chunks) {
            fFirstEntryPerChunk.push_back(next);
            next += chunk->length();
            fChunkIndex.push_back(next);
         }
         for (size_t si = 0, se = fValuesPtrPerSlot.size(); si != se; ++si) {
            fArrayVisitorPerSlot.push_back(ArrayPtrVisitor{fValuesPtrPerSlot.data() + si});
         }
      }

      /// This returns the ptr to the ptr to actual data.
      std::vector<void *> slotPtrs()
      {
         std::vector<void *> result;
         for (size_t i = 0; i < fValuesPtrPerSlot.size(); ++i) {
            result.push_back(fValuesPtrPerSlot.data() + i);
         }
         return result;
      }

      // Convenience method to avoid code duplication between
      // SetEntry and InitSlot
      void UncachedSlotLookup(unsigned int slot, ULong64_t entry)
      {
         // If entry is greater than the previous one,
         // we can skip all the chunks before the last one we
         // queried.
         size_t ci = 0;
         assert(slot < fLastChunkPerSlot.size());
         if (fLastEntryPerSlot[slot] < entry) {
            ci = fLastChunkPerSlot.at(slot);
         }

         for (size_t ce = fChunkIndex.size(); ci != ce; ++ci) {
            if (entry < fChunkIndex[ci]) {
               assert(slot < fLastChunkPerSlot.size());
               fLastChunkPerSlot[slot] = ci;
               break;
            }
         }

         // Update the pointer to the requested entry.
         // Notice that we need to find the entry
         auto chunk = fChunks.at(fLastChunkPerSlot[slot]);
         assert(slot < fArrayVisitorPerSlot.size());
         fArrayVisitorPerSlot[slot].SetEntry(entry - fFirstEntryPerChunk[fLastChunkPerSlot[slot]]);
         auto status = chunk->Accept(fArrayVisitorPerSlot.data() + slot);
         if (!status.ok()) {
            std::string msg = "Could not get pointer for slot ";
            msg += std::to_string(slot) + " looking at entry " + std::to_string(entry);
            throw std::runtime_error(msg);
         }
      }

      /// Set the current entry to be retrieved
      void SetEntry(unsigned int slot, ULong64_t entry)
      {
         // Same entry as before
         if (fLastEntryPerSlot[slot] == entry) {
            return;
         }
         UncachedSlotLookup(slot, entry);
      }
   };

   std::shared_ptr<arrow::Table> fTable;
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
   std::vector<std::string> fColumnNames;
   size_t fNSlots;

   std::vector<std::pair<size_t, size_t>> fGetterIndex; // (columnId, visitorId)
   std::vector<ValueGetter> fValueGetters;              // Visitors to be used to track and get entries. One per column.
   std::vector<void *> GetColumnReadersImpl(std::string_view name, const std::type_info &type) override;

public:
   TArrowDS(std::shared_ptr<arrow::Table> table, std::vector<std::string> const &columns);
   ~TArrowDS();
   const std::vector<std::string> &GetColumnNames() const override;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() override;
   std::string GetTypeName(std::string_view colName) const override;
   bool HasColumn(std::string_view colName) const override;
   void SetEntry(unsigned int slot, ULong64_t entry) override;
   void InitSlot(unsigned int slot, ULong64_t firstEntry) override;
   void SetNSlots(unsigned int nSlots) override;
   void Initialise() override;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create a Apache Arrow TDataFrame.
/// \param[in] table an apache::arrow table to use as a source.
TDataFrame MakeArrowDataFrame(std::shared_ptr<arrow::Table> table, std::vector<std::string> const &columns);

} // namespace TDF
} // namespace Experimental
} // namespace ROOT

#endif
