// Author: Giulio Eulisse CERN  2/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// clang-format off
/** \class ROOT::Experimental::TDF::TArrowDS
    \ingroup dataframe
    \brief TDataFrame data source class to interface with Apache Arrow.

The TArrowDS implements a proxy TDataSource to be able to use Apache Arrow
tables with TDataFrame.

A TDataFrame that adapts an arrow::Table class can be constructed using the factory method
ROOT::Experimental::TDF::MakeArrowDataFrame, which accepts one parameter:
1. An arrow::Table smart pointer.

The types of the columns are derived from the types in the associated
arrow::Schema. 

*/
// clang-format on

#include <ROOT/TDFUtils.hxx>
#include <ROOT/TSeq.hxx>
#include <ROOT/TArrowDS.hxx>
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
namespace Experimental {
namespace TDF {

/// Helper to get the contents of a given column

/// Helper to get the human readable name of type
class TDFTypeNameGetter : public ::arrow::TypeVisitor {
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
      fTypeName = "Long32_t";
      return arrow::Status::OK();
   }
   arrow::Status Visit(const arrow::UInt64Type &) override
   {
      fTypeName = "ULong64_t";
      return arrow::Status::OK();
   }
   arrow::Status Visit(const arrow::UInt32Type &) override
   {
      fTypeName = "ULong32_t";
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
   virtual arrow::Status Visit(const arrow::Int32Type &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::FloatType &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::DoubleType &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::StringType &) override { return arrow::Status::OK(); }
   virtual arrow::Status Visit(const arrow::BooleanType &) override { return arrow::Status::OK(); }

   using ::arrow::TypeVisitor::Visit;
};

////////////////////////////////////////////////////////////////////////
/// Constructor to create an Arrow TDataSource for TDataFrame.
/// \param[in] table the arrow Table to observe.
/// \param[in] columns the name of the columns to use
/// In case columns is empty, we use all the columns found in the table
TArrowDS::TArrowDS(std::shared_ptr<arrow::Table> inTable, std::vector<std::string> const &inColumns)
   : fTable{inTable}, fColumnNames{inColumns}, fNSlots(1)
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
      auto name = columnNames.front();
      auto index = table->schema()->GetFieldIndex(name);
      return table->column(index)->length();
   };

   // All columns are supposed to have the same number of entries.
   auto verifyColumnSize = [&table](std::shared_ptr<arrow::Column> column, int nRecords)
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
   SetNSlots(fNSlots);
}

////////////////////////////////////////////////////////////////////////
/// Destructor.
TArrowDS::~TArrowDS()
{
}

const std::vector<std::string> &TArrowDS::GetColumnNames() const
{
   return fColumnNames;
}

std::vector<std::pair<ULong64_t, ULong64_t>> TArrowDS::GetEntryRanges()
{
   auto entryRanges(std::move(fEntryRanges)); // empty fEntryRanges
   return entryRanges;
}

std::string TArrowDS::GetTypeName(std::string_view colName) const
{
   auto field = fTable->schema()->GetFieldByName(std::string(colName));
   if (!field) {
      std::string msg = "The dataset does not have column ";
      msg += colName;
      throw std::runtime_error(msg);
   }
   TDFTypeNameGetter typeGetter;
   auto status = field->type()->Accept(&typeGetter);
   if (status.ok() == false) {
      std::string msg = "TArrowDS does not support a column of type ";
      msg += field->type()->name();
      throw std::runtime_error(msg);
   }
   return typeGetter.result();
}

bool TArrowDS::HasColumn(std::string_view colName) const
{
   auto field = fTable->schema()->GetFieldByName(std::string(colName));
   if (!field) {
      return false;
   }
   return true;
}

void TArrowDS::SetEntry(unsigned int slot, ULong64_t entry)
{
   for (auto link : fGetterIndex) {
      auto column = fTable->column(link.first);
      auto &getter = fValueGetters[link.second];
      getter.SetEntry(slot, entry);
   }
}

void TArrowDS::InitSlot(unsigned int slot, ULong64_t entry)
{
   for (auto link : fGetterIndex) {
      auto column = fTable->column(link.first);
      auto &getter = fValueGetters[link.second];
      getter.UncachedSlotLookup(slot, entry);
   }
}

void TArrowDS::SetNSlots(unsigned int nSlots)
{
   // We dump all the previous getters structures and we rebuild it.
   auto nColumns = fGetterIndex.size();
   auto &outNSlots = fNSlots;
   auto &ranges = fEntryRanges;
   auto &table = fTable;
   auto &columnNames = fColumnNames;

   fValueGetters.clear();
   fValueGetters.reserve(nColumns);
   for (size_t ci = 0; ci != nColumns; ++ci) {
      auto chunkedArray = fTable->column(fGetterIndex[ci].first)->data();
      fValueGetters.push_back(ValueGetter{nSlots, chunkedArray->chunks()});
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
std::vector<void *> TArrowDS::GetColumnReadersImpl(std::string_view colName, const std::type_info &)
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
   assert(getterIdx < fValueGetters.size());
   return fValueGetters[getterIdx].slotPtrs();
}

void TArrowDS::Initialise()
{
}

/// Creates a TDataFrame using an arrow::Table as input.
/// \param[in] table the arrow Table to observe.
/// \param[in] columnNames the name of the columns to use
/// In case columnNames is empty, we use all the columns found in the table
TDataFrame MakeArrowDataFrame(std::shared_ptr<arrow::Table> table, std::vector<std::string> const &columnNames)
{
   ROOT::Experimental::TDataFrame tdf(std::make_unique<TArrowDS>(table, columnNames));
   return tdf;
}

} // namespace TDF
} // namespace Experimental
} // namespace ROOT
