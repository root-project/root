// Author: Enrico Guiraud, Danilo Piparo, Massimo Tumolo CERN  06/2018

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RCOLUMNREGISTER
#define ROOT_RDF_RCOLUMNREGISTER

#include <TString.h>

#include <algorithm>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Detail {
namespace RDF {
class RDefineBase;
class RLoopManager;
}
} // namespace Detail

namespace Internal {
namespace RDF {

namespace RDFDetail = ROOT::Detail::RDF;

class RVariationBase;

/**
 * \class ROOT::Internal::RDF::RColumnRegister
 * \ingroup dataframe
 * \brief A binder for user-defined columns and aliases.
 * The storage is copy-on-write and shared between all instances of the class that have the same values.
 */
class RColumnRegister {
   using ColumnNames_t = std::vector<std::string>;
   using DefinesMap_t = std::unordered_map<std::string, std::shared_ptr<RDFDetail::RDefineBase>>;
   /// See fVariations for more information on this type.
   using VariationsMap_t = std::unordered_multimap<std::string, std::shared_ptr<RVariationBase>>;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new name to the list returned by `GetNames` without booking a new column.
   ///
   /// This is needed because we abuse fColumnNames to also keep track of the aliases defined
   /// in each branch of the computation graph.
   /// Internally it recreates the vector with the new name, and swaps it with the old one.
   void AddName(std::string_view name);

private:
   // N.B. must come before fDefines, to be destructed after them
   std::shared_ptr<RDFDetail::RLoopManager> fLoopManager;

   /// Immutable map of Defines, can be shared among several nodes.
   /// When a new define is added (through a call to RInterface::Define or similar) a new map with the extra element is
   /// created.
   std::shared_ptr<const DefinesMap_t> fDefines;
   /// Immutable map of Aliases, can be shared among several nodes.
   std::shared_ptr<const std::unordered_map<std::string, std::string>> fAliases;
   /// Immutable multimap of Variations, can be shared among several nodes.
   /// The key is the name of an existing column, the values are all variations that affect that column.
   /// As a consequence, Variations that affect multiple columns are inserted multiple times, once per column.
   std::shared_ptr<const VariationsMap_t> fVariations;
   std::shared_ptr<const ColumnNames_t> fColumnNames; ///< Names of Defines and Aliases registered so far.

public:
   RColumnRegister(const RColumnRegister &) = default;
   RColumnRegister(RColumnRegister &&) = default;
   RColumnRegister &operator=(const RColumnRegister &) = default;

   RColumnRegister(std::shared_ptr<RDFDetail::RLoopManager> lm)
      : fLoopManager(lm), fDefines(std::make_shared<DefinesMap_t>()),
        fAliases(std::make_shared<std::unordered_map<std::string, std::string>>()),
        fVariations(std::make_shared<VariationsMap_t>()), fColumnNames(std::make_shared<ColumnNames_t>())
   {
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the list of the names of the defined columns (Defines + Aliases).
   ColumnNames_t GetNames() const { return *fColumnNames; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Returns a map of pointers to the defined columns.
   const DefinesMap_t &GetColumns() const { return *fDefines; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the multimap of systematic variations, see fVariations.
   const VariationsMap_t &GetVariations() const { return *fVariations; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Check if the provided name is tracked in the names list
   bool HasName(std::string_view name) const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new booked column.
   /// Internally it recreates the map with the new column, and swaps it with the old one.
   void AddColumn(const std::shared_ptr<RDFDetail::RDefineBase> &column);

   /// \brief Add a new alias to the ledger.
   void AddAlias(std::string_view alias, std::string_view colName);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return true if the given column name is an existing alias.
   bool IsAlias(const std::string &name) const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the actual column name that the alias resolves to.
   /// Drills through multiple levels of aliasing if needed.
   /// Returns the input in case it's not an alias.
   /// Expands `#%var` to `R_rdf_sizeof_var` (the #%var columns are implicitly-defined aliases).
   std::string ResolveAlias(std::string_view alias) const;

   /// \brief Register a new systematic variation.
   void AddVariation(const std::shared_ptr<RVariationBase> &variation);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Get the names of the variations that directly provide alternative values for this column.
   std::vector<std::string> GetVariationsFor(const std::string &column) const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Get the names of all variations that directly or indirectly affect a given column.
   ///
   /// This list includes variations applied to the column as well as variations applied to other
   /// columns on which the value of this column depends (typically via a Define expression).
   std::vector<std::string> GetVariationDeps(const std::string &column) const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Get the names of all variations that directly or indirectly affect the specified columns.
   ///
   /// This list includes variations applied to the columns as well as variations applied to other
   /// columns on which the value of any of these columns depend (typically via Define expressions).
   std::vector<std::string> GetVariationDeps(const ColumnNames_t &columns) const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the RVariation object that handles the specified variation of the specified column.
   RVariationBase &FindVariation(const std::string &colName, const std::string &variationName) const;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Empty the contents of this ledger.
   /// The only allowed operation on a RColumnRegister object after a call to Clear is its destruction.
   void Clear();
};

} // Namespace RDF
} // Namespace Internal
} // Namespace ROOT

#endif // ROOT_RDF_RCOLUMNREGISTER
