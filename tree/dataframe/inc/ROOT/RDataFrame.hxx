// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
  \defgroup dataframe Dataframe
ROOT's RDataFrame allows to analyse data stored in TTrees with a high level interface.
*/

#ifndef ROOT_RDATAFRAME
#define ROOT_RDATAFRAME

#include "TROOT.h" // To allow ROOT::EnableImplicitMT without including ROOT.h
#include "ROOT/RDF/RDatasetSpec.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "ROOT/RDF/Utils.hxx"
#include <string_view>
#include "RtypesCore.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

class TDirectory;
class TTree;

namespace ROOT {
namespace RDF {
class RDataSource;
}

namespace RDFDetail = ROOT::Detail::RDF;

class RDataFrame : public ROOT::RDF::RInterface<RDFDetail::RLoopManager> {
public:
   using ColumnNames_t = ROOT::RDF::ColumnNames_t;
   RDataFrame(std::string_view treeName, std::string_view filenameglob, const ColumnNames_t &defaultColumns = {});
   RDataFrame(std::string_view treename, const std::vector<std::string> &filenames,
              const ColumnNames_t &defaultColumns = {});
   RDataFrame(std::string_view treename, std::initializer_list<std::string> filenames,
              const ColumnNames_t &defaultColumns = {}):
              RDataFrame(treename, std::vector<std::string>(filenames), defaultColumns) {}
   RDataFrame(std::string_view treeName, ::TDirectory *dirPtr, const ColumnNames_t &defaultColumns = {});
   RDataFrame(TTree &tree, const ColumnNames_t &defaultColumns = {});
   RDataFrame(ULong64_t numEntries);
   RDataFrame(std::unique_ptr<ROOT::RDF::RDataSource>, const ColumnNames_t &defaultColumns = {});
   RDataFrame(ROOT::RDF::Experimental::RDatasetSpec spec);
};

namespace RDF {
namespace Experimental {

////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Factory method to create an RDataFrame from a JSON specification file.
/// \param[in] jsonFile Path of the JSON file, which should follow the format described in
///                     https://github.com/root-project/root/issues/11624
ROOT::RDataFrame FromSpec(const std::string &jsonFile);

} // namespace Experimental
} // namespace RDF

} // ns ROOT

/// Print a RDataFrame at the prompt
namespace cling {
std::string printValue(ROOT::RDataFrame *tdf);
} // ns cling

#endif // ROOT_RDATAFRAME
