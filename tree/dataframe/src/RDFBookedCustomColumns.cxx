#include "ROOT/RDFBookedCustomColumns.hxx"

namespace ROOT {
namespace Internal {
namespace RDF {

bool RBookedCustomColumns::HasName(std::string name) const
{
   return std::find(fCustomColumnsNames->begin(), fCustomColumnsNames->end(), name) != fCustomColumnsNames->end();
}

void RBookedCustomColumns::AddColumn(const std::shared_ptr<RDFDetail::RCustomColumnBase> &column,
                                     const std::string_view &name)
{
   auto newCols = std::make_shared<RCustomColumnBasePtrMap_t>(GetColumns());
   (*newCols)[std::string(name)] = column;
   fCustomColumns = newCols;
}

void RBookedCustomColumns::AddName(const std::string_view &name)
{
   auto newColsNames = std::make_shared<ColumnNames_t>(GetNames());
   newColsNames->emplace_back(name);
   fCustomColumnsNames = newColsNames;
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
