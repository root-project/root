// Author: Enrico Guiraud, Danilo Piparo, Massimo Tumolo CERN  06/2018
// Author: Vincenzo Eduardo Padulano CERN 05/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
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
#include <string_view>
#include <vector>
#include <utility>

namespace ROOT {
namespace RDF {
class RVariationsDescription;
}
namespace Detail {
namespace RDF {
class RColumnReaderBase;
class RDefineBase;
class RLoopManager;
}
} // namespace Detail

namespace Internal {
namespace RDF {

namespace RDFDetail = ROOT::Detail::RDF;

class RVariationBase;
class RVariationReader;
class RDefinesWithReaders;
class RVariationsWithReaders;

/**
 * \class ROOT::Internal::RDF::RColumnRegister
 * \ingroup dataframe
 * \brief A binder for user-defined columns, variations and aliases.
 *
 * The storage is copy-on-write and shared between all instances of the class that have the same values.
 *
 * Several components of an RDF computation graph make use of a column register. It keeps track of which columns have
 * been defined, varied or aliased at each point of the computation graph.
 * In many cases, the contents of the different column register instances are the same or only differ by a single
 * extra defined/varied/aliased column. For this reason, in order to avoid unnecessary data duplication, fDefines,
 * fAliases, fVariations and fColumnNames are all shared_ptr<const T> that (whenever possible) are shared across
 * RColumnRegister instances that are part of the same computation graph. If a new column, alias or variation is added
 * between one node and the next, then the new node contains a new instance of a RColumnRegister that shares all data
 * members with the previous instance except for the one data member that needed updating, which is replaced with a new
 * updated instance.
 *
 * The contents of the collections that keep track of other objects of the computation graph are not owned by the
 * RColumnRegister object. They are registered centrally by the RLoopManager and only accessed via reference in the
 * RColumnRegister.
 */
class RColumnRegister {
   using VariationsMap_t = std::unordered_multimap<std::string_view, ROOT::Internal::RDF::RVariationsWithReaders *>;
   using DefinesMap_t = std::vector<std::pair<std::string_view, ROOT::Internal::RDF::RDefinesWithReaders *>>;
   using AliasesMap_t = std::vector<std::pair<std::string_view, std::string_view>>;

   /// The head node of the computation graph this register belongs to. Never null.
   ROOT::Detail::RDF::RLoopManager *fLoopManager;

   /// Immutable multimap of Variations, can be shared among several nodes.
   /// The key is the name of an existing column, the values are all variations
   /// that affect that column. Variations that affect multiple columns are
   /// inserted in the map multiple times, once per column, and conversely each
   /// column (i.e. each key) can have several associated variations.
   std::shared_ptr<const VariationsMap_t> fVariations;
   /// Immutable collection of Defines, can be shared among several nodes.
   /// The pointee changes if a new Define node is added to the RColumnRegister.
   /// It is a vector because we rely on insertion order to recreate the branch
   /// of the computation graph where necessary.
   std::shared_ptr<const DefinesMap_t> fDefines;
   /// Immutable map of Aliases, can be shared among several nodes.
   /// The pointee changes if a new Alias node is added to the RColumnRegister.
   /// It is a vector because we rely on insertion order to recreate the branch
   /// of the computation graph where necessary.
   std::shared_ptr<const AliasesMap_t> fAliases;

   RVariationsWithReaders *FindVariationAndReaders(const std::string &colName, const std::string &variationName);

public:
   explicit RColumnRegister(ROOT::Detail::RDF::RLoopManager *lm);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the list of the names of the defined columns (Defines + Aliases).
   std::vector<std::string_view> GenerateColumnNames() const;

   std::vector<std::string_view> BuildDefineNames() const;

   RDFDetail::RDefineBase *GetDefine(std::string_view colName) const;

   bool IsDefineOrAlias(std::string_view name) const;

   void AddDefine(std::shared_ptr<RDFDetail::RDefineBase> column);

   void AddAlias(std::string_view alias, std::string_view colName);

   bool IsAlias(std::string_view name) const;
   bool IsDefine(std::string_view name) const;

   std::string_view ResolveAlias(std::string_view alias) const;

   void AddVariation(std::shared_ptr<RVariationBase> variation);

   std::vector<std::string> GetVariationsFor(const std::string &column) const;

   std::vector<std::string> GetVariationDeps(const std::string &column) const;

   std::vector<std::string> GetVariationDeps(const std::vector<std::string> &columns) const;

   ROOT::RDF::RVariationsDescription BuildVariationsDescription() const;

   RDFDetail::RColumnReaderBase *GetReader(unsigned int slot, const std::string &colName,
                                           const std::string &variationName, const std::type_info &tid);

   RDFDetail::RColumnReaderBase *
   GetReaderUnchecked(unsigned int slot, const std::string &colName, const std::string &variationName);
};

} // Namespace RDF
} // Namespace Internal
} // Namespace ROOT

#endif // ROOT_RDF_RCOLUMNREGISTER
