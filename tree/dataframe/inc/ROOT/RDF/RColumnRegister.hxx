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

class RDefineReader;
class RVariationBase;
class RVariationReader;

/// A helper type that keeps track of RDefine objects and their corresponding RDefineReaders.
class RDefineAndReaders {
   using RDefineBase = RDFDetail::RDefineBase;

   std::shared_ptr<RDefineBase> fDefine;
   // Column readers per variation (in the map) per slot (in the vector).
   std::vector<std::unordered_map<std::string, std::shared_ptr<RDefineReader>>> fReadersPerVariation;

public:
   RDefineAndReaders(std::shared_ptr<RDefineBase> define, unsigned int nSlots);
   RDefineBase *GetDefine() const { return fDefine.get(); }
   std::shared_ptr<RDefineReader>
   GetReader(unsigned int slot, const std::string &variationName, const std::type_info &);
};

class RVariationAndReaders {
   std::shared_ptr<RVariationBase> fVariation;
   // Column readers for this RVariation for a given variation (map key) and a given slot (vector element).
   std::vector<std::unordered_map<std::string, std::shared_ptr<RVariationReader>>> fReadersPerVariation;

public:
   RVariationAndReaders(std::shared_ptr<RVariationBase> variation, unsigned int nSlots);
   RVariationBase *GetVariation() const { return fVariation.get(); }
   std::shared_ptr<RVariationReader> GetReader(unsigned int slot, const std::string &colName,
                                               const std::string &variationName, const std::type_info &tid);
};

/**
 * \class ROOT::Internal::RDF::RColumnRegister
 * \ingroup dataframe
 * \brief A binder for user-defined columns and aliases.
 * The storage is copy-on-write and shared between all instances of the class that have the same values.
 */
class RColumnRegister {
   using ColumnNames_t = std::vector<std::string>;
   using DefinesMap_t = std::unordered_map<std::string, std::shared_ptr<RDefineAndReaders>>;
   /// See fVariations for more information on this type.
   using VariationsMap_t = std::unordered_multimap<std::string, std::shared_ptr<RVariationAndReaders>>;

   std::shared_ptr<RDFDetail::RLoopManager> fLoopManager;

   /// Immutable collection of Defines, can be shared among several nodes.
   /// The pointee changes if a new Define node is added to the RColumnRegister.
   std::shared_ptr<DefinesMap_t> fDefines;
   /// Immutable map of Aliases, can be shared among several nodes.
   std::shared_ptr<const std::unordered_map<std::string, std::string>> fAliases;
   /// Immutable multimap of Variations, can be shared among several nodes.
   /// The key is the name of an existing column, the values are all variations that affect that column.
   /// Variations that affect multiple columns are inserted in the map multiple times, once per column,
   /// and conversely each column (i.e. each key) can have several associated variations.
   std::shared_ptr<VariationsMap_t> fVariations;
   std::shared_ptr<const ColumnNames_t> fColumnNames; ///< Names of Defines and Aliases registered so far.

   void AddName(std::string_view name);

   RVariationAndReaders *FindVariationAndReaders(const std::string &colName, const std::string &variationName);

public:
   RColumnRegister(const RColumnRegister &) = default;
   RColumnRegister(RColumnRegister &&) = default;
   RColumnRegister &operator=(const RColumnRegister &) = default;

   explicit RColumnRegister(std::shared_ptr<RDFDetail::RLoopManager> lm)
      : fLoopManager(lm), fDefines(std::make_shared<DefinesMap_t>()),
        fAliases(std::make_shared<std::unordered_map<std::string, std::string>>()),
        fVariations(std::make_shared<VariationsMap_t>()), fColumnNames(std::make_shared<ColumnNames_t>())
   {
   }
   ~RColumnRegister();

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Return the list of the names of the defined columns (Defines + Aliases).
   ColumnNames_t GetNames() const { return *fColumnNames; }

   ColumnNames_t BuildDefineNames() const;

   RDFDetail::RDefineBase *GetDefine(const std::string &colName) const;

   bool IsDefineOrAlias(std::string_view name) const;

   void AddDefine(std::shared_ptr<RDFDetail::RDefineBase> column);

   void AddAlias(std::string_view alias, std::string_view colName);

   bool IsAlias(const std::string &name) const;

   std::string ResolveAlias(std::string_view alias) const;

   void AddVariation(std::shared_ptr<RVariationBase> variation);

   std::vector<std::string> GetVariationsFor(const std::string &column) const;

   std::vector<std::string> GetVariationDeps(const std::string &column) const;

   std::vector<std::string> GetVariationDeps(const ColumnNames_t &columns) const;

   ROOT::RDF::RVariationsDescription BuildVariationsDescription() const;

   std::shared_ptr<RDFDetail::RColumnReaderBase> GetReader(unsigned int slot, const std::string &colName,
                                                           const std::string &variationName, const std::type_info &tid);
};

} // Namespace RDF
} // Namespace Internal
} // Namespace ROOT

#endif // ROOT_RDF_RCOLUMNREGISTER
