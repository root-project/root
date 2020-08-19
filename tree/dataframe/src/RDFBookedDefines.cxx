#include "ROOT/RDF/RBookedDefines.hxx"

namespace ROOT {
namespace Internal {
namespace RDF {

bool RBookedDefines::HasName(std::string_view name) const
{
   const auto ccolnamesEnd = fDefinesNames->end();
   return ccolnamesEnd != std::find(fDefinesNames->begin(), ccolnamesEnd, name);
}

void RBookedDefines::AddColumn(const std::shared_ptr<RDFDetail::RDefineBase> &column, std::string_view name)
{
   auto newCols = std::make_shared<RDefineBasePtrMap_t>(GetColumns());
   (*newCols)[std::string(name)] = column;
   fDefines = newCols;
}

void RBookedDefines::AddName(std::string_view name)
{
   auto newColsNames = std::make_shared<ColumnNames_t>(GetNames());
   newColsNames->emplace_back(std::string(name));
   fDefinesNames = newColsNames;
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
