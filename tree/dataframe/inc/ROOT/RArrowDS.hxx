/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RARROWDS
#define ROOT_RARROWDS

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

   std::vector<std::pair<size_t, size_t>> fGetterIndex; // (columnId, visitorId)
   std::vector<std::unique_ptr<ROOT::Internal::RDF::TValueGetter>> fValueGetters; // Visitors to be used to track and get entries. One per column.
   std::vector<void *> GetColumnReadersImpl(std::string_view name, const std::type_info &type) final;

public:
   RArrowDS(std::shared_ptr<arrow::Table> table, std::vector<std::string> const &columns);
   // Rule of five
   RArrowDS(const RArrowDS &) = delete;
   RArrowDS &operator=(const RArrowDS &) = delete;
   RArrowDS(RArrowDS &&) = delete;
   RArrowDS &operator=(RArrowDS &&) = delete;
   ~RArrowDS() final;

   const std::vector<std::string> &GetColumnNames() const final;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final;
   std::string GetTypeName(std::string_view colName) const final;
   bool HasColumn(std::string_view colName) const final;
   bool SetEntry(unsigned int slot, ULong64_t entry) final;
   void InitSlot(unsigned int slot, ULong64_t firstEntry) final;
   void SetNSlots(unsigned int nSlots) final;
   void Initialize() final;
   std::string GetLabel() final;
};

RDataFrame FromArrow(std::shared_ptr<arrow::Table> table, std::vector<std::string> const &columnNames);

} // namespace RDF

} // namespace ROOT

#endif
