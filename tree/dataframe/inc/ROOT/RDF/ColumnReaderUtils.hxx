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
GetColumnReader(unsigned int slot, std::shared_ptr<RColumnReaderBase> defineOrVariationReader, RLoopManager &lm,
                TTreeReader *r, const std::string &colName)
{
   if (defineOrVariationReader != nullptr)
      return defineOrVariationReader;

   // Check if we already inserted a reader for this column in the dataset column readers (RDataSource or Tree/TChain
   // readers)
   auto datasetColReader = lm.GetDatasetColumnReader(slot, colName, typeid(T));
   if (datasetColReader != nullptr)
      return datasetColReader;

   assert(r != nullptr && "We could not find a reader for this column, this should never happen at this point.");

   // Make a RTreeColumnReader for this column and insert it in RLoopManager's map
   auto treeColReader = std::make_unique<RTreeColumnReader<T>>(*r, colName);
   return lm.AddTreeColumnReader(slot, colName, std::move(treeColReader), typeid(T));
}

/// This type aggregates some of the arguments passed to GetColumnReaders.
/// We need to pass a single RColumnReadersInfo object rather than each argument separately because with too many
/// arguments passed, gcc 7.5.0 and cling disagree on the ABI, which leads to the last function argument being read
/// incorrectly from a compiled GetColumnReaders symbols when invoked from a jitted symbol.
struct RColumnReadersInfo {
   const std::vector<std::string> &fColNames;
   RColumnRegister &fColRegister;
   const bool *fIsDefine;
   RLoopManager &fLoopManager;
};

/// Create a group of column readers, one per type in the parameter pack.
template <typename... ColTypes>
std::array<std::shared_ptr<RDFDetail::RColumnReaderBase>, sizeof...(ColTypes)>
GetColumnReaders(unsigned int slot, TTreeReader *r, TypeList<ColTypes...>, const RColumnReadersInfo &colInfo,
                 const std::string &variationName = "nominal")
{
   // see RColumnReadersInfo for why we pass these arguments like this rather than directly as function arguments
   const auto &colNames = colInfo.fColNames;
   auto &lm = colInfo.fLoopManager;
   auto &colRegister = colInfo.fColRegister;

   int i = -1;
   std::array<std::shared_ptr<RDFDetail::RColumnReaderBase>, sizeof...(ColTypes)> ret{
      {{(++i, GetColumnReader<ColTypes>(slot, colRegister.GetReader(slot, colNames[i], variationName, typeid(ColTypes)),
                                        lm, r, colNames[i]))}...}};
   return ret;
}

// Shortcut overload for the case of no columns
inline std::array<std::shared_ptr<RDFDetail::RColumnReaderBase>, 0>
GetColumnReaders(unsigned int, TTreeReader *, TypeList<>, const RColumnReadersInfo &, const std::string & = "nominal")
{
   return {};
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RDF_COLUMNREADERS
