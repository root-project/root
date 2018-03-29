#ifndef ROOT_TARROWTDS
#define ROOT_TARROWTDS

#include "ROOT/TDataFrame.hxx"
#include "ROOT/TDataSource.hxx"

#include <memory>

namespace arrow {
   class Table;
}

namespace ROOT {
namespace Internal {
namespace TDF {
   class ValueGetter;
} // namespace TDF
} // namespace Internal

namespace Experimental {
namespace TDF {

class TArrowDS final : public TDataSource {
private:
   std::shared_ptr<arrow::Table> fTable;
   std::vector<std::pair<ULong64_t, ULong64_t>> fEntryRanges;
   std::vector<std::string> fColumnNames;
   size_t fNSlots = 0U;

   std::vector<std::pair<size_t, size_t>> fGetterIndex; // (columnId, visitorId)
   std::vector<std::unique_ptr<ROOT::Internal::TDF::ValueGetter>> fValueGetters; // Visitors to be used to track and get entries. One per column.
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
