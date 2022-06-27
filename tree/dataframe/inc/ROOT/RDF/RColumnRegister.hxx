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

   std::shared_ptr<RDFDetail::RLoopManager> fLoopManager;

   /// Immutable map of Defines, can be shared among several nodes.
   /// When a new define is added (through a call to RInterface::Define or similar) a new map with the extra element is
   /// created.
   std::shared_ptr<const DefinesMap_t> fDefines;
   /// Immutable map of Aliases, can be shared among several nodes.
   std::shared_ptr<const std::unordered_map<std::string, std::string>> fAliases;
   /// Immutable multimap of Variations, can be shared among several nodes.
   /// The key is the name of an existing column, the values are all variations that affect that column.
   /// Variations that affect multiple columns are inserted in the map multiple times, once per column,
   /// and conversely each column (i.e. each key) can have several associated variations.
   std::shared_ptr<const VariationsMap_t> fVariations;
   std::shared_ptr<const ColumnNames_t> fColumnNames; ///< Names of Defines and Aliases registered so far.

   void AddName(std::string_view name);

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
   ~RColumnRegister();

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the list of the names of the defined columns (Defines + Aliases).
   ColumnNames_t GetNames() const { return *fColumnNames; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return a map of pointers to the defined columns.
   const DefinesMap_t &GetDefines() const { return *fDefines; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the multimap of systematic variations, see fVariations.
   const VariationsMap_t &GetVariations() const { return *fVariations; }

   bool IsDefineOrAlias(std::string_view name) const;

   void AddDefine(const std::shared_ptr<RDFDetail::RDefineBase> &column);

   void AddAlias(std::string_view alias, std::string_view colName);

   bool IsAlias(const std::string &name) const;

   std::string ResolveAlias(std::string_view alias) const;

   void AddVariation(const std::shared_ptr<RVariationBase> &variation);

   std::vector<std::string> GetVariationsFor(const std::string &column) const;

   std::vector<std::string> GetVariationDeps(const std::string &column) const;

   std::vector<std::string> GetVariationDeps(const ColumnNames_t &columns) const;

   RVariationBase &FindVariation(const std::string &colName, const std::string &variationName) const;
};

} // Namespace RDF
} // Namespace Internal
} // Namespace ROOT

#endif // ROOT_RDF_RCOLUMNREGISTER
