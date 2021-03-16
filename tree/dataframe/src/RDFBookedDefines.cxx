/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
   const std::string colName(name);
   (*newCols)[colName] = column;
   fDefines = newCols;
   AddName(colName);
}

void RBookedDefines::AddName(std::string_view name)
{
   const auto &names = GetNames();
   if (std::find(names.begin(), names.end(), name) != names.end())
      return; // must be a Redefine of an existing column. Nothing to do.

   auto newColsNames = std::make_shared<ColumnNames_t>(names);
   newColsNames->emplace_back(std::string(name));
   fDefinesNames = newColsNames;
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
