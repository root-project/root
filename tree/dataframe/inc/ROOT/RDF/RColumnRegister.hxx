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
}
} // namespace Detail

namespace Internal {
namespace RDF {

namespace RDFDetail = ROOT::Detail::RDF;

/**
 * \class ROOT::Internal::RDF::RColumnRegister
 * \ingroup dataframe
 * \brief A binder for user-defined columns and aliases.
 * The storage is copy-on-write and shared between all instances of the class that have the same values.
 */
class RColumnRegister {
   using RDefineBasePtrMap_t = std::unordered_map<std::string, std::shared_ptr<RDFDetail::RDefineBase>>;
   using ColumnNames_t = std::vector<std::string>;

   // Since RColumnRegister is meant to be an immutable, copy-on-write object, the actual values are set as const
   using RDefineBasePtrMapPtr_t = std::shared_ptr<const RDefineBasePtrMap_t>;
   using ColumnNamesPtr_t = std::shared_ptr<const ColumnNames_t>;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new name to the list returned by `GetNames` without booking a new column.
   ///
   /// This is needed because we abuse fColumnNames to also keep track of the aliases defined
   /// in each branch of the computation graph.
   /// Internally it recreates the vector with the new name, and swaps it with the old one.
   void AddName(std::string_view name);

private:
   /// Immutable map of Defines, can be shared among several nodes.
   /// When a new define is added (through a call to RInterface::Define or similar) a new map with the extra element is
   /// created.
   RDefineBasePtrMapPtr_t fDefines;
   /// Immutable map of Aliases, can be shared among several nodes.
   std::shared_ptr<const std::unordered_map<std::string, std::string>> fAliases;
   ColumnNamesPtr_t fColumnNames; ///< Names of Defines and Aliases registered so far.

public:
   RColumnRegister(const RColumnRegister &) = default;
   RColumnRegister(RColumnRegister &&) = default;
   RColumnRegister &operator=(const RColumnRegister &) = default;

   RColumnRegister()
      : fDefines(std::make_shared<RDefineBasePtrMap_t>()),
        fAliases(std::make_shared<std::unordered_map<std::string, std::string>>()),
        fColumnNames(std::make_shared<ColumnNames_t>())
   {
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the list of the names of the defined columns (Defines + Aliases)
   ColumnNames_t GetNames() const { return *fColumnNames; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Returns the list of the pointers to the defined columns
   const RDefineBasePtrMap_t &GetColumns() const { return *fDefines; }

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

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Empty the contents of this ledger.
   /// The only allowed operation on a RColumnRegister object after a call to Clear is its destruction.
   void Clear();
};

} // Namespace RDF
} // Namespace Internal
} // Namespace ROOT

#endif // ROOT_RDF_RCOLUMNREGISTER
