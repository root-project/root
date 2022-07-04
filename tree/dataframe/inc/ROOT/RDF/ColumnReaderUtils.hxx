// Author: Enrico Guiraud CERN 09/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_COLUMNREADERUTILS
#define ROOT_RDF_COLUMNREADERUTILS

#include "RColumnReaderBase.hxx"
#include "RColumnRegister.hxx"
#include "RDefineBase.hxx"
#include "RDefineReader.hxx"
#include "RDSColumnReader.hxx"
#include "RLoopManager.hxx"
#include "RTreeColumnReader.hxx"
#include "RVariationBase.hxx"
#include "RVariationReader.hxx"

#include <ROOT/RDataSource.hxx>
#include <ROOT/TypeTraits.hxx>
#include <TTreeReader.h>

#include <array>
#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <typeinfo> // for typeid
#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {

using namespace ROOT::TypeTraits;
namespace RDFDetail = ROOT::Detail::RDF;

template <typename T>
std::shared_ptr<RDFDetail::RColumnReaderBase>
MakeColumnReader(unsigned int slot, RDefineBase *define, RLoopManager &lm, TTreeReader *r, const std::string &colName,
                 RVariationBase *variation, const std::string &variationName)
{
   using Ret_t = std::shared_ptr<RDFDetail::RColumnReaderBase>;

   // variations have precedence over everything else: if this is not null, it means we are in the
   // universe where this variation applies.
   if (variation != nullptr)
      return Ret_t{new RVariationReader(slot, colName, variationName, *variation, typeid(T))};

   // defines come second, so that Redefine'd columns have precedence over dataset columns
   if (define != nullptr) {
      if (variationName != "nominal" && IsStrInVec(variationName, define->GetVariations()))
         define = &define->GetVariedDefine(variationName);
      return Ret_t{new RDefineReader(slot, *define, typeid(T))};
   }

   // Check if we already inserted a reader for this column in the dataset column readers (RDataSource or Tree/TChain
   // readers)
   auto datasetColReader = lm.GetDatasetColumnReader(slot, colName, typeid(T));
   if (datasetColReader != nullptr)
      return datasetColReader;

   assert(r != nullptr && "We could not find a reader for this column, this should never happen at this point.");

   // Make a RTreeColumnReader for this column and insert it in RLoopManager's map
   auto treeColReader = std::make_unique<RTreeColumnReader<T>>(*r, colName);
   lm.AddTreeColumnReader(slot, colName, std::move(treeColReader), typeid(T));
   return lm.GetDatasetColumnReader(slot, colName, typeid(T));
}

/// This type aggregates some of the arguments passed to MakeColumnReaders.
/// We need to pass a single RColumnReadersInfo object rather than each argument separately because with too many
/// arguments passed, gcc 7.5.0 and cling disagree on the ABI, which leads to the last function argument being read
/// incorrectly from a compiled MakeColumnReaders symbols when invoked from a jitted symbol.
struct RColumnReadersInfo {
   const std::vector<std::string> &fColNames;
   const RColumnRegister &fColRegister;
   const bool *fIsDefine;
   RLoopManager &fLoopManager;
};

/// Create a group of column readers, one per type in the parameter pack.
template <typename... ColTypes>
std::array<std::shared_ptr<RDFDetail::RColumnReaderBase>, sizeof...(ColTypes)>
MakeColumnReaders(unsigned int slot, TTreeReader *r, TypeList<ColTypes...>, const RColumnReadersInfo &colInfo,
                  const std::string &variationName = "nominal")
{
   // see RColumnReadersInfo for why we pass these arguments like this rather than directly as function arguments
   const auto &colNames = colInfo.fColNames;
   auto &lm = colInfo.fLoopManager;
   const auto &colRegister = colInfo.fColRegister;

   // the i-th element indicates whether variation variationName provides alternative values for the i-th column
   std::array<bool, sizeof...(ColTypes)> doesVariationApply;
   if (variationName == "nominal")
      doesVariationApply.fill(false);
   else {
      for (auto i = 0u; i < sizeof...(ColTypes); ++i)
         doesVariationApply[i] = IsStrInVec(variationName, colRegister.GetVariationsFor(colNames[i]));
   }

   int i = -1;
   std::array<std::shared_ptr<RDFDetail::RColumnReaderBase>, sizeof...(ColTypes)> ret{
      {{(++i, MakeColumnReader<ColTypes>(slot, colRegister.GetDefine(colNames[i]), lm, r, colNames[i],
                                         doesVariationApply[i] ? &colRegister.FindVariation(colNames[i], variationName)
                                                               : nullptr,
                                         variationName))}...}};
   return ret;

   // avoid bogus "unused variable" warnings
   (void)slot;
   (void)r;
}

// Shortcut overload for the case of no columns
inline std::array<std::shared_ptr<RDFDetail::RColumnReaderBase>, 0>
MakeColumnReaders(unsigned int, TTreeReader *, TypeList<>, const RColumnReadersInfo &, const std::string & = "nominal")
{
   return {};
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RDF_COLUMNREADERS
