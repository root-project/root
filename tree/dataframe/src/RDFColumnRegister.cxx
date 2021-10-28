/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RColumnRegister.hxx"
#include "ROOT/RDF/RDefineBase.hxx"

namespace ROOT {
namespace Internal {
namespace RDF {

bool RColumnRegister::HasName(std::string_view name) const
{
   const auto ccolnamesEnd = fColumnNames->end();
   return ccolnamesEnd != std::find(fColumnNames->begin(), ccolnamesEnd, name);
}

void RColumnRegister::AddColumn(const std::shared_ptr<RDFDetail::RDefineBase> &column)
{
   auto newDefines = std::make_shared<RDefineBasePtrMap_t>(GetColumns());
   const std::string &colName = column->GetName();
   (*newDefines)[colName] = column;
   fDefines = std::move(newDefines);
   AddName(colName);
}

void RColumnRegister::AddName(std::string_view name)
{
   const auto &names = GetNames();
   if (std::find(names.begin(), names.end(), name) != names.end())
      return; // must be a Redefine of an existing column. Nothing to do.

   auto newColsNames = std::make_shared<ColumnNames_t>(names);
   newColsNames->emplace_back(std::string(name));
   fColumnNames = newColsNames;
}

void RColumnRegister::AddAlias(std::string_view alias, std::string_view colName)
{
   // at this point validation of alias and colName has already happened, we trust that
   // this is a new, valid alias.
   auto newAliases = std::make_shared<std::unordered_map<std::string, std::string>>(*fAliases);
   (*newAliases)[std::string(alias)] = ResolveAlias(colName);
   fAliases = std::move(newAliases);
   AddName(alias);
}

bool RColumnRegister::IsAlias(const std::string &name) const
{
   return fAliases->find(name) != fAliases->end();
}

std::string RColumnRegister::ResolveAlias(std::string_view alias) const
{
   std::string aliasStr{alias};

   // #var is an alias for R_rdf_sizeof_var
   if (aliasStr.size() > 1 && aliasStr[0] == '#')
      return "R_rdf_sizeof_" + aliasStr.substr(1);

   auto it = fAliases->find(aliasStr);
   if (it != fAliases->end())
      return it->second;

   return aliasStr; // not an alias, i.e. already resolved
}

void RColumnRegister::Clear()
{
   fDefines.reset();
   fColumnNames.reset();
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
