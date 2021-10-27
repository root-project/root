// Author: Enrico Guiraud, Danilo Piparo, Massimo Tumolo CERN  06/2018

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFBOOKEDCUSTOMCOLUMNS
#define ROOT_RDFBOOKEDCUSTOMCOLUMNS

#include <TString.h>

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Detail {
namespace RDF {
class RDefineBase;
}
}

namespace Internal {
namespace RDF {

namespace RDFDetail = ROOT::Detail::RDF;

/**
 * \class ROOT::Internal::RDF::RBookedDefines
 * \ingroup dataframe
 * \brief A binder for user-defined columns and aliases.
 * The storage is copy-on-write and shared between all instances of the class that have the same values.
 */
class RBookedDefines {
   using RDefineBasePtrMap_t = std::map<std::string, std::shared_ptr<RDFDetail::RDefineBase>>;
   using ColumnNames_t = std::vector<std::string>;

   // Since RBookedDefines is meant to be an immutable, copy-on-write object, the actual values are set as const
   using RDefineBasePtrMapPtr_t = std::shared_ptr<const RDefineBasePtrMap_t>;
   using ColumnNamesPtr_t = std::shared_ptr<const ColumnNames_t>;

private:
   /// Immutable map of Defines, can be shared among several nodes.
   /// When a new define is added (through a call to RInterface::Define or similar) a new map with the extra element is
   /// created.
   RDefineBasePtrMapPtr_t fDefines;
   ColumnNamesPtr_t fColumnNames;  ///< Names of Defines and Aliases registered so far.

public:
   RBookedDefines(const RBookedDefines &) = default;
   RBookedDefines(RBookedDefines &&) = default;
   RBookedDefines &operator=(const RBookedDefines &) = default;

   RBookedDefines(RDefineBasePtrMapPtr_t defines, ColumnNamesPtr_t defineNames)
      : fDefines(defines), fColumnNames(defineNames)
   {
   }

   RBookedDefines()
      : fDefines(std::make_shared<RDefineBasePtrMap_t>()),
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
   void AddColumn(const std::shared_ptr<RDFDetail::RDefineBase> &column, std::string_view name);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Add a new name to the list returned by `GetNames` without booking a new column.
   ///
   /// This is needed because we abuse fColumnNames to also keep track of the aliases defined
   /// in each branch of the computation graph.
   /// Internally it recreates the vector with the new name, and swaps it with the old one.
   void AddName(std::string_view name);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Empty the contents of this ledger.
   /// The only allowed operation on a RBookedDefines object after a call to Clear is its destruction.
   void Clear();
};

} // Namespace RDF
} // Namespace Internal
} // Namespace ROOT

#endif
