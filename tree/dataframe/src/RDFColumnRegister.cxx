/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RColumnRegister.hxx"
#include "ROOT/RDF/RDefineBase.hxx"
#include "ROOT/RDF/RDefineReader.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RVariationBase.hxx"
#include "ROOT/RDF/RVariationsDescription.hxx"
#include "ROOT/RDF/RVariationReader.hxx"
#include "ROOT/RDF/Utils.hxx" // IsStrInVec

#include <cassert>
#include <set>

namespace ROOT {
namespace Internal {
namespace RDF {

RDefineAndReaders::RDefineAndReaders(std::shared_ptr<RDefineBase> define, unsigned int nSlots)
   : fDefine(std::move(define)), fReadersPerVariation(nSlots)
{
   assert(fDefine != nullptr);
}

std::shared_ptr<RDFInternal::RDefineReader>
RDefineAndReaders::GetReader(unsigned int slot, const std::string &variationName, const std::type_info &tid)
{
   auto &defineReaders = fReadersPerVariation[slot];

   auto it = defineReaders.find(variationName);
   if (it != defineReaders.end())
      return it->second;

   auto *define = fDefine.get();
   if (variationName != "nominal")
      define = &define->GetVariedDefine(variationName);

   auto insertion = defineReaders.insert({variationName, std::make_shared<RDefineReader>(slot, *define, tid)});
   return insertion.first->second;
}

RVariationAndReaders::RVariationAndReaders(std::shared_ptr<RVariationBase> variation, unsigned int nSlots)
   : fVariation(std::move(variation)), fReadersPerVariation(nSlots)
{
   assert(fVariation != nullptr);
}

////////////////////////////////////////////////////////////////////////////
/// Return a column reader for the given slot, column and variation, or a nullptr if not available.
std::shared_ptr<RVariationReader> RVariationAndReaders::GetReader(unsigned int slot, const std::string &colName,
                                                                  const std::string &variationName,
                                                                  const std::type_info &tid)
{
   assert(IsStrInVec(variationName, fVariation->GetVariationNames())); // otherwise we should not be here

   auto &varReaders = fReadersPerVariation[slot];

   auto it = varReaders.find(variationName);
   if (it != varReaders.end())
      return it->second;

   auto insertion = varReaders.insert(
      {variationName, std::make_shared<RVariationReader>(slot, colName, variationName, *fVariation, tid)});
   return insertion.first->second;
}

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

////////////////////////////////////////////////////////////////////////////
/// \brief Return the list of the names of defined columns (no aliases).
ColumnNames_t RColumnRegister::BuildDefineNames() const
{
   ColumnNames_t names;
   names.reserve(fDefines->size());
   for (auto &kv : *fDefines) {
      names.emplace_back(kv.first);
   }
   return names;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Return the RDefine for the requested column name, or nullptr.
RDFDetail::RDefineBase *RColumnRegister::GetDefine(const std::string &colName) const
{
   auto it = fDefines->find(colName);
   return it == fDefines->end() ? nullptr : it->second->GetDefine();
}

////////////////////////////////////////////////////////////////////////////
/// \brief Check if the provided name is tracked in the names list
bool RColumnRegister::IsDefineOrAlias(std::string_view name) const
{
   const auto ccolnamesEnd = fColumnNames->end();
   return ccolnamesEnd != std::find(fColumnNames->begin(), ccolnamesEnd, name);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a new defined column.
/// Internally it recreates the map with the new column, and swaps it with the old one.
void RColumnRegister::AddDefine(std::shared_ptr<RDFDetail::RDefineBase> define)
{
   auto newDefines = std::make_shared<DefinesMap_t>(*fDefines);
   const std::string &colName = define->GetName();

   // this will assign over a pre-existing element in case this AddDefine is due to a Redefine
   (*newDefines)[colName] = std::make_shared<RDefineAndReaders>(define, fLoopManager->GetNSlots());

   fDefines = std::move(newDefines);
   AddName(colName);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Register a new systematic variation.
void RColumnRegister::AddVariation(std::shared_ptr<RVariationBase> variation)
{
   auto newVariations = std::make_shared<VariationsMap_t>(*fVariations);
   const std::vector<std::string> &colNames = variation->GetColumnNames();
   for (auto &colName : colNames)
      newVariations->insert({colName, std::make_shared<RVariationAndReaders>(variation, fLoopManager->GetNSlots())});
   fVariations = std::move(newVariations);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get the names of the variations that directly provide alternative values for this column.
std::vector<std::string> RColumnRegister::GetVariationsFor(const std::string &column) const
{
   std::vector<std::string> variations;
   auto range = fVariations->equal_range(column);
   for (auto it = range.first; it != range.second; ++it)
      for (const auto &variationName : it->second->GetVariation()->GetVariationNames())
         variations.emplace_back(variationName);

   return variations;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get the names of all variations that directly or indirectly affect a given column.
///
/// This list includes variations applied to the column as well as variations applied to other
/// columns on which the value of this column depends (typically via a Define expression).
std::vector<std::string> RColumnRegister::GetVariationDeps(const std::string &column) const
{
   return GetVariationDeps(std::vector<std::string>{column});
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get the names of all variations that directly or indirectly affect the specified columns.
///
/// This list includes variations applied to the columns as well as variations applied to other
/// columns on which the value of any of these columns depend (typically via Define expressions).
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
         for (const auto &v : defineIt->second->GetDefine()->GetVariations())
            variationNames.insert(v);
      }
   }

   return {variationNames.begin(), variationNames.end()};
}

////////////////////////////////////////////////////////////////////////////
/// \brief Return the RVariationAndReaders object that handles the specified variation of the specified column, or null.
RVariationAndReaders *
RColumnRegister::FindVariationAndReaders(const std::string &colName, const std::string &variationName)
{
   auto range = fVariations->equal_range(colName);
   if (range.first == fVariations->end())
      return nullptr;
   for (auto it = range.first; it != range.second; ++it) {
      if (IsStrInVec(variationName, it->second->GetVariation()->GetVariationNames()))
         return it->second.get();
   }

   return nullptr;
}

ROOT::RDF::RVariationsDescription RColumnRegister::BuildVariationsDescription() const
{
   std::set<const RVariationBase *> uniqueVariations;
   for (auto &e : *fVariations)
      uniqueVariations.insert(e.second->GetVariation());

   const std::vector<const RVariationBase *> variations(uniqueVariations.begin(), uniqueVariations.end());
   return ROOT::RDF::RVariationsDescription{variations};
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a new name to the list returned by `GetNames` without booking a new column.
///
/// This is needed because we abuse fColumnNames to also keep track of the aliases defined
/// in each branch of the computation graph.
/// Internally it recreates the vector with the new name, and swaps it with the old one.
void RColumnRegister::AddName(std::string_view name)
{
   const auto &names = *fColumnNames;
   if (std::find(names.begin(), names.end(), name) != names.end())
      return; // must be a Redefine of an existing column. Nothing to do.

   auto newColsNames = std::make_shared<ColumnNames_t>(names);
   newColsNames->emplace_back(std::string(name));
   fColumnNames = newColsNames;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a new alias to the ledger.
void RColumnRegister::AddAlias(std::string_view alias, std::string_view colName)
{
   // at this point validation of alias and colName has already happened, we trust that
   // this is a new, valid alias.
   auto newAliases = std::make_shared<std::unordered_map<std::string, std::string>>(*fAliases);
   (*newAliases)[std::string(alias)] = ResolveAlias(colName);
   fAliases = std::move(newAliases);
   AddName(alias);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Return true if the given column name is an existing alias.
bool RColumnRegister::IsAlias(const std::string &name) const
{
   return fAliases->find(name) != fAliases->end();
}

////////////////////////////////////////////////////////////////////////////
/// \brief Return the actual column name that the alias resolves to.
/// Drills through multiple levels of aliasing if needed.
/// Returns the input in case it's not an alias.
/// Expands `#%var` to `R_rdf_sizeof_var` (the #%var columns are implicitly-defined aliases).
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

std::shared_ptr<RDFDetail::RColumnReaderBase> RColumnRegister::GetReader(unsigned int slot, const std::string &colName,
                                                                         const std::string &variationName,
                                                                         const std::type_info &requestedType)
{
   // try variations first
   if (variationName != "nominal") {
      auto *variationAndReaders = FindVariationAndReaders(colName, variationName);
      if (variationAndReaders != nullptr)
         return variationAndReaders->GetReader(slot, colName, variationName, requestedType);
   }

   // otherwise try defines
   auto it = fDefines->find(colName);
   if (it != fDefines->end())
      return it->second->GetReader(slot, variationName, requestedType);

   return nullptr;
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT
