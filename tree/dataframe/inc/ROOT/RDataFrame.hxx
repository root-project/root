// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
  \defgroup dataframe DataFrame
ROOT's RDataFrame allows to analyse data stored in TTrees with a high level interface.
*/

#ifndef ROOT_RDATAFRAME
#define ROOT_RDATAFRAME

#include <sstream> // std::ostringstream
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "ROOT/RCutFlowReport.hxx"
#include "ROOT/RDFInterface.hxx"
#include "ROOT/RDFNodes.hxx"
#include "ROOT/RDFUtils.hxx"
#include "ROOT/RDataSource.hxx"
#include "ROOT/TypeTraits.hxx"
#include "ROOT/RStringView.hxx"
#include "RtypesCore.h"
#include "TTree.h"

class TDirectory;

namespace ROOT {
namespace RDFDetail = ROOT::Detail::RDF;
namespace RDFInternal = ROOT::Internal::RDF;
namespace TTraits = ROOT::TypeTraits;

class RDataFrame : public ROOT::RDF::RInterface<RDFDetail::RLoopManager> {

public:
   using ColumnNames_t = RDFDetail::ColumnNames_t;
   using RDataSource = ROOT::RDF::RDataSource;
   RDataFrame(std::string_view treeName, std::string_view filenameglob, const ColumnNames_t &defaultBranches = {});
   RDataFrame(std::string_view treename, const std::vector<std::string> &filenames,
              const ColumnNames_t &defaultBranches = {});
   RDataFrame(std::string_view treeName, ::TDirectory *dirPtr, const ColumnNames_t &defaultBranches = {});
   RDataFrame(TTree &tree, const ColumnNames_t &defaultBranches = {});
   RDataFrame(ULong64_t numEntries);
   RDataFrame(std::unique_ptr<RDataSource>, const ColumnNames_t &defaultBranches = {});
};

} // end NS ROOT

/// Print a RDataFrame at the prompt
namespace cling {
std::string printValue(ROOT::RDataFrame *tdf);
} // namespace cling

#endif // ROOT_TDATAFRAME
