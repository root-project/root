#include "ROOT/RDF/RBookedCustomColumns.hxx"

namespace ROOT {
namespace Internal {
namespace RDF {

bool RBookedCustomColumns::HasName(std::string_view name) const
{
   const auto ccolnamesEnd = fCustomColumnsNames->end();
   return ccolnamesEnd != std::find(fCustomColumnsNames->begin(), ccolnamesEnd, name);
}

void RBookedCustomColumns::AddColumn(const std::shared_ptr<RDFDetail::RCustomColumnBase> &column,
                                     std::string_view name)
{
   auto newCols = std::make_shared<RCustomColumnBasePtrMap_t>(GetColumns());
   (*newCols)[std::string(name)] = column;
   fCustomColumns = newCols;
}

void RBookedCustomColumns::AddName(std::string_view name)
{
   auto newColsNames = std::make_shared<ColumnNames_t>(GetNames());
   newColsNames->emplace_back(std::string(name));
   fCustomColumnsNames = newColsNames;
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
