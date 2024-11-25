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

RColumnRegister::RColumnRegister(ROOT::Detail::RDF::RLoopManager *lm)
   : fLoopManager(lm),
     fVariations(std::make_shared<const VariationsMap_t>()),
     fDefines(std::make_shared<const DefinesMap_t>()),
     fAliases(std::make_shared<const AliasesMap_t>())
{
   // TODO: The RLoopManager could be taken by reference.
   // We cannot do it now because it would prevent RColumRegister from
   // being copiable, and that is currently done.
   assert(fLoopManager != nullptr);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Return the list of the names of defined columns (no aliases).
std::vector<std::string_view> RColumnRegister::BuildDefineNames() const
{
   std::vector<std::string_view> names;
   names.reserve(fDefines->size());
   for (const auto &kv : *fDefines) {
      names.push_back(kv.first);
   }
   return names;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Return the RDefine for the requested column name, or nullptr.
RDFDetail::RDefineBase *RColumnRegister::GetDefine(std::string_view colName) const
{
   auto it = std::find_if(fDefines->begin(), fDefines->end(),
                          [&colName](const DefinesMap_t::value_type &kv) { return kv.first == colName; });
   return it == fDefines->end() ? nullptr : &it->second->GetDefine();
}

////////////////////////////////////////////////////////////////////////////
/// \brief Check if the provided name is tracked in the names list
bool RColumnRegister::IsDefineOrAlias(std::string_view name) const
{
   return IsDefine(name) || IsAlias(name);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a new defined column.
/// Registers the pair (columnName, columnReader) with the current RDataFrame,
/// then keeps a reference to the inserted objects to keep track of the
/// available columns for this node. Internally it recreates the collection with
/// the new column, and swaps it with the old one.
void RColumnRegister::AddDefine(std::shared_ptr<RDFDetail::RDefineBase> define)
{
   const auto colIt = fLoopManager->GetColumnNamesCache().Insert(define->GetName());
   auto insertion_defs = fLoopManager->GetUniqueDefinesWithReaders().insert(
      {*colIt, std::make_unique<ROOT::Internal::RDF::RDefinesWithReaders>(define, fLoopManager->GetNSlots(),
                                                                          fLoopManager->GetColumnNamesCache())});

   auto newDefines = std::make_shared<DefinesMap_t>(*fDefines);
   // Structured bindings cannot be used in lambda captures (until C++20)
   // so we explicitly define variable names here
   const auto &colAndDefIt = insertion_defs.first;
   // If there is a Redefine, we need to reinsert the new pointer to the readers into the same element
   if (auto previousDefIt =
          std::find_if(newDefines->begin(), newDefines->end(),
                       [&colAndDefIt](const DefinesMap_t::value_type &kv) { return kv.first == colAndDefIt->first; });
       previousDefIt != newDefines->end()) {
      previousDefIt->second = colAndDefIt->second.get();
   } else {
      newDefines->push_back({colAndDefIt->first, colAndDefIt->second.get()});
   }
   fDefines = std::move(newDefines);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Register a new systematic variation.
void RColumnRegister::AddVariation(std::shared_ptr<RVariationBase> variation)
{
   auto newVariations = std::make_shared<VariationsMap_t>(*fVariations);
   const std::vector<std::string> &colNames = variation->GetColumnNames();

   // Cache column names for this variation and store views for later use
   for (const auto &colName : colNames) {
      auto colIt = fLoopManager->GetColumnNamesCache().Insert(colName);
      auto [colAndVariationsIt, _2] = fLoopManager->GetUniqueVariationsWithReaders().insert(
         {*colIt, std::make_unique<ROOT::Internal::RDF::RVariationsWithReaders>(variation, fLoopManager->GetNSlots())});
      newVariations->insert({colAndVariationsIt->first, colAndVariationsIt->second.get()});
   }

   fVariations = std::move(newVariations);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Get the names of the variations that directly provide alternative values for this column.
std::vector<std::string> RColumnRegister::GetVariationsFor(const std::string &column) const
{
   std::vector<std::string> variations;
   auto range = fVariations->equal_range(column);
   for (auto it = range.first; it != range.second; ++it)
      for (const auto &variationName : it->second->GetVariation().GetVariationNames())
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
std::vector<std::string> RColumnRegister::GetVariationDeps(const std::vector<std::string> &columns) const
{
   // here we assume that columns do not contain aliases, they must have already been resolved
   std::set<std::string> variationNames;

   for (const auto &col : columns) {
      const auto &variations = GetVariationsFor(col);
      for (const auto &var : variations)
         variationNames.insert(var);

      // For Define'd columns, add the systematic variations they depend on to the set
      auto defineIt = std::find_if(fDefines->begin(), fDefines->end(),
                                   [&col](const DefinesMap_t::value_type &kv) { return kv.first == col; });
      if (defineIt != fDefines->end()) {
         for (const auto &v : defineIt->second->GetDefine().GetVariations())
            variationNames.insert(v);
      }
   }

   return {variationNames.begin(), variationNames.end()};
}

////////////////////////////////////////////////////////////////////////////
/// \brief Return the RVariationsWithReaders object that handles the specified variation of the specified column, or
/// null.
RVariationsWithReaders *
RColumnRegister::FindVariationAndReaders(const std::string &colName, const std::string &variationName)
{
   auto range = fVariations->equal_range(colName);
   if (range.first == fVariations->end())
      return nullptr;
   for (auto it = range.first; it != range.second; ++it) {
      if (IsStrInVec(variationName, it->second->GetVariation().GetVariationNames()))
         return it->second;
   }

   return nullptr;
}

ROOT::RDF::RVariationsDescription RColumnRegister::BuildVariationsDescription() const
{
   std::set<const RVariationBase *> uniqueVariations;
   for (auto &e : *fVariations)
      uniqueVariations.insert(&e.second->GetVariation());

   const std::vector<const RVariationBase *> variations(uniqueVariations.begin(), uniqueVariations.end());
   return ROOT::RDF::RVariationsDescription{variations};
}

////////////////////////////////////////////////////////////////////////////
/// \brief Add a new alias to the ledger.
/// Registers the strings alias, colName with the current RDataFrame, then uses
/// references to those string to create the new pair for the collection of
/// aliases of this node.
void RColumnRegister::AddAlias(std::string_view alias, std::string_view colName)
{
   // at this point validation of alias and colName has already happened, we trust that
   // this is a new, valid alias.
   auto &colNamesCache = fLoopManager->GetColumnNamesCache();
   auto aliasIt = colNamesCache.Insert(std::string(alias));
   auto colIt = colNamesCache.Insert(std::string(colName));

   auto newAliases = std::make_shared<AliasesMap_t>(*fAliases);
   // If an alias was already present we need to substitute it with the new one
   if (auto previousAliasIt =
          std::find_if(newAliases->begin(), newAliases->end(),
                       [&aliasIt](const AliasesMap_t::value_type &kv) { return kv.first == *aliasIt; });
       previousAliasIt != newAliases->end()) {
      previousAliasIt->second = ResolveAlias(*colIt);
   } else {
      newAliases->push_back({*aliasIt, ResolveAlias(*colIt)});
   }

   fAliases = std::move(newAliases);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Return true if the given column name is an existing alias.
bool RColumnRegister::IsAlias(std::string_view name) const
{
   return std::find_if(fAliases->begin(), fAliases->end(),
                       [&name](const AliasesMap_t::value_type &kv) { return kv.first == name; }) != fAliases->end();
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Return true if the given column name is an existing defined column.
bool RColumnRegister::IsDefine(std::string_view name) const
{
   return std::find_if(fDefines->begin(), fDefines->end(),
                       [&name](const DefinesMap_t::value_type &kv) { return kv.first == name; }) != fDefines->end();
}

////////////////////////////////////////////////////////////////////////////
/// \brief Return the actual column name that the alias resolves to.
/// Drills through multiple levels of aliasing if needed.
/// Returns the input in case it's not an alias.
/// Expands `#%var` to `R_rdf_sizeof_var` (the #%var columns are implicitly-defined aliases).
std::string_view RColumnRegister::ResolveAlias(std::string_view alias) const
{

   // #var is an alias for R_rdf_sizeof_var
   if (alias.size() > 1 && alias[0] == '#') {
      std::string sizeof_colname{"R_rdf_sizeof_"};
      sizeof_colname.append(alias.substr(1));
      auto colIt = fLoopManager->GetColumnNamesCache().Insert(sizeof_colname);
      return *colIt;
   }

   if (auto it = std::find_if(fAliases->begin(), fAliases->end(),
                              [&alias](const AliasesMap_t::value_type &kv) { return kv.first == alias; });
       it != fAliases->end())
      return it->second;

   return alias; // not an alias, i.e. already resolved
}

/// Return a RDefineReader or a RVariationReader, or nullptr if not available.
/// If requestedType does not match the actual type of the Define or Variation, an exception is thrown.
RDFDetail::RColumnReaderBase *RColumnRegister::GetReader(unsigned int slot, const std::string &colName,
                                                         const std::string &variationName,
                                                         const std::type_info &requestedType)
{
   // try variations first
   if (variationName != "nominal") {
      auto *variationAndReaders = FindVariationAndReaders(colName, variationName);
      if (variationAndReaders != nullptr) {
         const auto &actualType = variationAndReaders->GetVariation().GetTypeId();
         CheckReaderTypeMatches(actualType, requestedType, colName);
         return &variationAndReaders->GetReader(slot, colName, variationName);
      }
   }

   // otherwise try defines
   auto it = std::find_if(fDefines->begin(), fDefines->end(),
                          [&colName](const DefinesMap_t::value_type &kv) { return kv.first == colName; });
   if (it != fDefines->end()) {
      const auto &actualType = it->second->GetDefine().GetTypeId();
      CheckReaderTypeMatches(actualType, requestedType, colName);
      return &it->second->GetReader(slot, variationName);
   }

   return nullptr;
}

/// Return a RDefineReader or a RVariationReader, or nullptr if not available.
/// No type checking is done on the requested reader.
RDFDetail::RColumnReaderBase *
RColumnRegister::GetReaderUnchecked(unsigned int slot, const std::string &colName, const std::string &variationName)
{
   // try variations first
   if (variationName != "nominal") {
      if (auto *variationAndReaders = FindVariationAndReaders(colName, variationName)) {
         return &variationAndReaders->GetReader(slot, colName, variationName);
      }
   }

   // otherwise try defines
   if (auto it =
          std::find_if(fDefines->begin(), fDefines->end(), [&colName](const auto &kv) { return kv.first == colName; });
       it != fDefines->end()) {
      return &it->second->GetReader(slot, variationName);
   }

   return nullptr;
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT

std::vector<std::string_view> ROOT::Internal::RDF::RColumnRegister::GenerateColumnNames() const
{
   std::vector<std::string_view> ret;
   ret.reserve(fDefines->size() + fAliases->size());
   for (const auto &kv : *fDefines)
      ret.push_back(kv.first);
   for (const auto &kv : *fAliases)
      ret.push_back(kv.first);
   return ret;
}
