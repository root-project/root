/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RColumnRegister.hxx"
#include "ROOT/RDF/RDefineBase.hxx"
#include "ROOT/RDF/RVariationBase.hxx"
#include "ROOT/RDF/Utils.hxx" // IsStrInVec

#include <cassert>
#include <set>

namespace ROOT {
namespace Internal {
namespace RDF {

RColumnRegister::~RColumnRegister()
{
   // Explicitly empty the containers to decrement the reference count of the
   // various shared_ptr's, which might cause destructors to be called. For
   // this, we need the RLoopManager to stay around and we do not want to rely
   // on the order during member destruction.
   fAliases.reset();
   fDefines.reset();
   fVariations.reset();
   fColumnNames.reset();
}

bool RColumnRegister::HasName(std::string_view name) const
{
   const auto ccolnamesEnd = fColumnNames->end();
   return ccolnamesEnd != std::find(fColumnNames->begin(), ccolnamesEnd, name);
}

void RColumnRegister::AddColumn(const std::shared_ptr<RDFDetail::RDefineBase> &column)
{
   auto newDefines = std::make_shared<DefinesMap_t>(*fDefines);
   const std::string &colName = column->GetName();
   (*newDefines)[colName] = column;
   fDefines = std::move(newDefines);
   AddName(colName);
}

void RColumnRegister::AddVariation(const std::shared_ptr<RVariationBase> &variation)
{
   auto newVariations = std::make_shared<VariationsMap_t>(*fVariations);
   const std::vector<std::string> &colNames = variation->GetColumnNames();
   for (auto &colName : colNames)
      newVariations->insert({colName, variation});
   fVariations = std::move(newVariations);
}

std::vector<std::string> RColumnRegister::GetVariationsFor(const std::string &column) const
{
   std::vector<std::string> variations;
   auto range = fVariations->equal_range(column);
   for (auto it = range.first; it != range.second; ++it)
      for (const auto &variationName : it->second->GetVariationNames())
         variations.emplace_back(variationName);

   return variations;
}

std::vector<std::string> RColumnRegister::GetVariationDeps(const std::string &column) const
{
   return GetVariationDeps(std::vector<std::string>{column});
}

std::vector<std::string> RColumnRegister::GetVariationDeps(const ColumnNames_t &columns) const
{
   // here we assume that columns do not contain aliases, they must have already been resolved
   std::set<std::string> variationNames;

   for (const auto &col : columns) {
      const auto &variations = GetVariationsFor(col);
      for (const auto &var : variations)
         variationNames.insert(var);

      // For Define'd columns, add the systematic variations they depend on to the set
      auto defineIt = fDefines->find(col);
      if (defineIt != fDefines->end()) {
         for (const auto &v : defineIt->second->GetVariations())
            variationNames.insert(v);
      }
   }

   return {variationNames.begin(), variationNames.end()};
}

/// \brief Return the RVariation object that handles the specified variation of the specified column.
RVariationBase &RColumnRegister::FindVariation(const std::string &colName, const std::string &variationName) const
{
   auto range = fVariations->equal_range(colName);
   assert(range.first != fVariations->end() && "Could not find the variation you asked for. This should never happen.");
   auto it = range.first;
   while (it != fVariations->end() && !IsStrInVec(variationName, it->second->GetVariationNames()))
      ++it;
   assert(it != fVariations->end() && "Could not find the variation you asked for. This should never happen.");
   return *it->second;
}

void RColumnRegister::AddName(std::string_view name)
{
   const auto &names = *fColumnNames;
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

} // namespace RDF
} // namespace Internal
} // namespace ROOT
