// Author: Giulio Eulisse CERN  2/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// clang-format off
/** \class ROOT::RDF::RArrowDS
    \ingroup dataframe
    \brief RDataFrame data source class to interface with Apache Arrow.

The RArrowDS implements a proxy RDataSource to be able to use Apache Arrow
tables with RDataFrame.

A RDataFrame that adapts an arrow::Table class can be constructed using the factory method
ROOT::RDF::MakeArrowDataFrame, which accepts one parameter:
1. An arrow::Table smart pointer.

The types of the columns are derived from the types in the associated
arrow::Schema.

*/
// clang-format on

#include <ROOT/RDFUtils.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/RArrowDS.hxx>
#include <ROOT/RMakeUnique.hxx>

#include <algorithm>
#include <sstream>
#include <string>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <arrow/table.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif


namespace ROOT {
namespace Internal {
namespace RDF {
// Per slot visitor of an Array.
class ArrayPtrVisitor : public ::arrow::ArrayVisitor {
private:
   /// The pointer to update.
   void **fResult;
   bool fCachedBool{false};   // Booleans need to be unpacked, so we use a cached entry.
   std::string fCachedString;
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
class TValueGetter {
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
   TValueGetter(size_t slots, arrow::ArrayVector chunks)
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
   std::vector<void *> SlotPtrs()
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

} // namespace RDF
} // namespace Internal


namespace RDF {

/// Helper to get the contents of a given column

/// Helper to get the human readable name of type
class RDFTypeNameGetter : public ::arrow::TypeVisitor {
private:
   std::string fTypeName;

public:
   arrow::Status Visit(const arrow::Int64Type &) override
   {
      fTypeName = "Long64_t";
      return arrow::Status::OK();
   }
   arrow::Status Visit(const arrow::Int32Type &) override
   {
      fTypeName = "Long_t";
      return arrow::Status::OK();
   }
   arrow::Status Visit(const arrow::UInt64Type &) override
   {
      fTypeName = "ULong64_t";
      return arrow::Status::OK();
   }
   arrow::Status Visit(const arrow::UInt32Type &) override
   {
      fTypeName = "ULong_t";
      return arrow::Status::OK();
   }
   arrow::Status Visit(const arrow::FloatType &) override
   {
      fTypeName = "float";
      return arrow::Status::OK();
   }
   arrow::Status Visit(const arrow::DoubleType &) override
   {
      fTypeName = "double";
      return arrow::Status::OK();
   }
   arrow::Status Visit(const arrow::StringType &) override
   {
      fTypeName = "string";
      return arrow::Status::OK();
   }
   arrow::Status Visit(const arrow::BooleanType &) override
   {
      fTypeName = "bool";
      return arrow::Status::OK();
   }
   std::string result() { return fTypeName; }

   using ::arrow::TypeVisitor::Visit;
};

/// Helper to determine if a given Column is a supported type.
class VerifyValidColumnType : public ::arrow::TypeVisitor {
private:
public:
   virtual arrow::Status Visit(const arrow::Int64Type &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::UInt64Type &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::Int32Type &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::UInt32Type &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::FloatType &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::DoubleType &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::StringType &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::BooleanType &) override { return arrow::Status::OK(); }

   using ::arrow::TypeVisitor::Visit;
};




////////////////////////////////////////////////////////////////////////
/// Constructor to create an Arrow RDataSource for RDataFrame.
/// \param[in] table the arrow Table to observe.
/// \param[in] columns the name of the columns to use
/// In case columns is empty, we use all the columns found in the table
RArrowDS::RArrowDS(std::shared_ptr<arrow::Table> inTable, std::vector<std::string> const &inColumns)
   : fTable{inTable}, fColumnNames{inColumns}
{
   auto &columnNames = fColumnNames;
   auto &table = fTable;
   auto &index = fGetterIndex;
   // We want to allow people to specify which columns they
   // need so that we can think of upfront IO optimizations.
   auto filterWantedColumns = [&columnNames, &table]()
   {
      if (columnNames.empty()) {
         for (auto &field : table->schema()->fields()) {
            columnNames.push_back(field->name());
         }
      }
   };

   auto getRecordsFirstColumn = [&columnNames, &table]()
   {
      if (columnNames.empty()) {
         throw std::runtime_error("At least one column required");
      }
      const auto name = columnNames.front();
      const auto columnIdx = table->schema()->GetFieldIndex(name);
      return table->column(columnIdx)->length();
   };

   // All columns are supposed to have the same number of entries.
   auto verifyColumnSize = [](std::shared_ptr<arrow::Column> column, int nRecords)
   {
      if (column->length() != nRecords) {
         std::string msg = "Column ";
         msg += column->name() + " has a different number of entries.";
         throw std::runtime_error(msg);
      }
   };

   /// For the moment we support only a few native types.
   auto verifyColumnType = [](std::shared_ptr<arrow::Column> column) {
      auto verifyType = std::make_unique<VerifyValidColumnType>();
      auto result = column->type()->Accept(verifyType.get());
      if (result.ok() == false) {
         std::string msg = "Column ";
         msg += column->name() + " contains an unsupported type.";
         throw std::runtime_error(msg);
      }
   };

   /// This is used to create an index between the columnId
   /// and the associated getter.
   auto addColumnToGetterIndex = [&index](int columnId)
   {
      index.push_back(std::make_pair(columnId, index.size()));
   };

   /// Assuming we can get called more than once, we need to
   /// reset the getter index each time.
   auto resetGetterIndex = [&index]() { index.clear(); };

   /// This is what initialization actually does
   filterWantedColumns();
   resetGetterIndex();
   auto nRecords = getRecordsFirstColumn();
   for (auto &columnName : fColumnNames) {
      auto columnIdx = fTable->schema()->GetFieldIndex(columnName);
      addColumnToGetterIndex(columnIdx);

      auto column = fTable->column(columnIdx);
      verifyColumnSize(column, nRecords);
      verifyColumnType(column);
   }
}

////////////////////////////////////////////////////////////////////////
/// Destructor.
RArrowDS::~RArrowDS()
{
}

const std::vector<std::string> &RArrowDS::GetColumnNames() const
{
   return fColumnNames;
}

std::vector<std::pair<ULong64_t, ULong64_t>> RArrowDS::GetEntryRanges()
{
   auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
   return entryRanges;
}

std::string RArrowDS::GetTypeName(std::string_view colName) const
{
   auto field = fTable->schema()->GetFieldByName(std::string(colName));
   if (!field) {
      std::string msg = "The dataset does not have column ";
      msg += colName;
      throw std::runtime_error(msg);
   }
   RDFTypeNameGetter typeGetter;
   auto status = field->type()->Accept(&typeGetter);
   if (status.ok() == false) {
      std::string msg = "RArrowDS does not support a column of type ";
      msg += field->type()->name();
      throw std::runtime_error(msg);
   }
   return typeGetter.result();
}

bool RArrowDS::HasColumn(std::string_view colName) const
{
   auto field = fTable->schema()->GetFieldByName(std::string(colName));
   if (!field) {
      return false;
   }
   return true;
}

bool RArrowDS::SetEntry(unsigned int slot, ULong64_t entry)
{
   for (auto link : fGetterIndex) {
      auto column = fTable->column(link.first);
      auto &getter = fValueGetters[link.second];
      getter->SetEntry(slot, entry);
   }
   return true;
}

void RArrowDS::InitSlot(unsigned int slot, ULong64_t entry)
{
   for (auto link : fGetterIndex) {
      auto column = fTable->column(link.first);
      auto &getter = fValueGetters[link.second];
      getter->UncachedSlotLookup(slot, entry);
   }
}

void RArrowDS::SetNSlots(unsigned int nSlots)
{
   assert(0U == fNSlots && "Setting the number of slots even if the number of slots is different from zero.");

   // We dump all the previous getters structures and we rebuild it.
   auto nColumns = fGetterIndex.size();
   auto &outNSlots = fNSlots;
   auto &ranges = fEntryRanges;
   auto &table = fTable;
   auto &columnNames = fColumnNames;

   fValueGetters.clear();
   for (size_t ci = 0; ci != nColumns; ++ci) {
      auto chunkedArray = fTable->column(fGetterIndex[ci].first)->data();
      fValueGetters.emplace_back(std::make_unique<ROOT::Internal::RDF::TValueGetter>(nSlots, chunkedArray->chunks()));
   }

   // We use the same logic as the ROOTDS.
   auto splitInEqualRanges = [&outNSlots, &ranges](int nRecords, unsigned int newNSlots)
   {
      ranges.clear();
      outNSlots = newNSlots;
      const auto chunkSize = nRecords / outNSlots;
      const auto remainder = 1U == outNSlots ? 0 : nRecords % outNSlots;
      auto start = 0UL;
      auto end = 0UL;
      for (auto i : ROOT::TSeqU(outNSlots)) {
         start = end;
         end += chunkSize;
         ranges.emplace_back(start, end);
         (void)i;
      }
      ranges.back().second += remainder;
   };

   auto getNRecords = [&table, &columnNames]()->int
   {
      auto index = table->schema()->GetFieldIndex(columnNames.front());
      return table->column(index)->length();
   };

   auto nRecords = getNRecords();
   splitInEqualRanges(nRecords, nSlots);
}

/// This needs to return a pointer to the pointer each value getter
/// will point to.
std::vector<void *> RArrowDS::GetColumnReadersImpl(std::string_view colName, const std::type_info &)
{
   auto &index = fGetterIndex;
   auto findGetterIndex = [&index](unsigned int column)
   {
      for (auto &entry : index) {
         if (entry.first == column) {
            return entry.second;
         }
      }
      throw std::runtime_error("No column found at index " + std::to_string(column));
   };

   const int columnIdx = fTable->schema()->GetFieldIndex(std::string(colName));
   const int getterIdx = findGetterIndex(columnIdx);
   assert(getterIdx != -1);
   assert((unsigned int)getterIdx < fValueGetters.size());
   return fValueGetters[getterIdx]->SlotPtrs();
}

void RArrowDS::Initialise()
{
}

std::string RArrowDS::GetDataSourceType()
{
   return "ArrowDS";
}

/// Creates a RDataFrame using an arrow::Table as input.
/// \param[in] table the arrow Table to observe.
/// \param[in] columnNames the name of the columns to use
/// In case columnNames is empty, we use all the columns found in the table
RDataFrame MakeArrowDataFrame(std::shared_ptr<arrow::Table> table, std::vector<std::string> const &columnNames)
{
   ROOT::RDataFrame tdf(std::make_unique<RArrowDS>(table, columnNames));
   return tdf;
}

} // namespace RDF

} // namespace ROOT
