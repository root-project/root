/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RARROWTDS
#define ROOT_RARROWTDS

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDataSource.hxx"

#include <memory>

namespace arrow {
class Table;
}

namespace ROOT {
namespace Internal {
namespace RDF {
class TValueGetter;
} // namespace RDF
} // namespace Internal

namespace RDF {

class RArrowDS final : public RDataSource {
private:
   std::shared_ptr<arrow::Table> fTable;
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
   std::vector<std::string> fColumnNames;
   size_t fNSlots = 0U;

   std::vector<std::pair<size_t, size_t>> fGetterIndex; // (columnId, visitorId)
   std::vector<std::unique_ptr<ROOT::Internal::RDF::TValueGetter>> fValueGetters; // Visitors to be used to track and get entries. One per column.
   std::vector<void *> GetColumnReadersImpl(std::string_view name, const std::type_info &type) override;

public:
   RArrowDS(std::shared_ptr<arrow::Table> table, std::vector<std::string> const &columns);
   ~RArrowDS();
   const std::vector<std::string> &GetColumnNames() const override;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() override;
   std::string GetTypeName(std::string_view colName) const override;
   bool HasColumn(std::string_view colName) const override;
   bool SetEntry(unsigned int slot, ULong64_t entry) override;
   void InitSlot(unsigned int slot, ULong64_t firstEntry) override;
   void SetNSlots(unsigned int nSlots) override;
   void Initialize() override;
   std::string GetLabel() override;
};

////////////////////////////////////////////////////////////////////////////////////////////////
RDataFrame MakeArrowDataFrame(std::shared_ptr<arrow::Table> table, std::vector<std::string> const &columnNames);

} // namespace RDF

} // namespace ROOT

#endif
