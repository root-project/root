// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
  \defgroup dataframe Data Frame
The ROOT Data Frame allows to analyse data stored in TTrees with a high level interface.
*/

#ifndef ROOT_TDATAFRAME
#define ROOT_TDATAFRAME

#include "ROOT/TActionResultProxy.hxx"
#include "ROOT/TDataFrameInterface.hxx"
#include "ROOT/TDFNodes.hxx"
#include "ROOT/TDFUtils.hxx"
#include "TChain.h"

#include <memory>
#include <iosfwd> // std::ostringstream
#include <stdexcept>
#include <string>
class TDirectory;
class TTree;

#ifndef __ROOTCLING__
R__LOAD_LIBRARY(libTreePlayer)
#endif

namespace cling {
// TDataFrame pretty-printing
std::string printValue(ROOT::Experimental::TDataFrame *tdf);
}

namespace ROOT {
namespace Experimental {

class TDataFrame : public TDataFrameInterface<ROOT::Detail::TDataFrameImpl> {
private:
   std::shared_ptr<TTree> fTree;
   void InitTree(TTree &tree, bool ownsTree);

public:
   TDataFrame(const std::string &treeName, const std::string &filenameglob, const BranchNames_t &defaultBranches = {});
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Build the dataframe
   /// \tparam FILENAMESCOLL The type of the file collection: only requirement: must have begin and end.
   /// \param[in] treeName Name of the tree contained in the directory
   /// \param[in] filenamescoll Collection of file names, for example a list of strings.
   /// \param[in] defaultBranches Collection of default branches.
   ///
   /// The default branches are looked at in case no branch is specified in the
   /// booking of actions or transformations.
   /// See ROOT::Experimental::TDataFrameInterface for the documentation of the
   /// methods available.
   template <
      typename FILENAMESCOLL,
      typename std::enable_if<ROOT::Internal::TDFTraitsUtils::TIsContainer<FILENAMESCOLL>::fgValue, int>::type = 0>
   TDataFrame(const std::string &treeName, const FILENAMESCOLL &filenamescoll,
              const BranchNames_t &defaultBranches = {});
   TDataFrame(const std::string &treeName, ::TDirectory *dirPtr, const BranchNames_t &defaultBranches = {});
   TDataFrame(TTree &tree, const BranchNames_t &defaultBranches = {});
};

template <typename FILENAMESCOLL,
          typename std::enable_if<ROOT::Internal::TDFTraitsUtils::TIsContainer<FILENAMESCOLL>::fgValue, int>::type>
TDataFrame::TDataFrame(const std::string &treeName, const FILENAMESCOLL &filenamescoll,
                       const BranchNames_t &defaultBranches)
   : TDataFrameInterface<ROOT::Detail::TDataFrameImpl>(
        std::make_shared<ROOT::Detail::TDataFrameImpl>(nullptr, defaultBranches))
{
   auto chain = new TChain(treeName.c_str());
   for (auto &fileName : filenamescoll) chain->Add(ROOT::Internal::ToConstCharPtr(fileName));
   fTree = std::make_shared<TTree>(static_cast<TTree *>(chain));
   fProxiedPtr->SetTree(chain);
}

} // end NS Experimental
} // end NS ROOT

////////////////////////////////////////////////////////////////////////////////
/// Print a TDataFrame at the prompt:
namespace cling {
inline std::string printValue(ROOT::Experimental::TDataFrame *tdf)
{
   auto df          = tdf->GetDataFrameChecked();
   auto treeName    = df->GetTreeName();
   auto defBranches = df->GetDefaultBranches();
   auto tmpBranches = df->GetTmpBranches();

   std::ostringstream ret;
   ret << "A data frame built on top of the " << treeName << " dataset.";
   if (!defBranches.empty()) {
      if (defBranches.size() == 1)
         ret << "\nDefault branch: " << defBranches[0];
      else {
         ret << "\nDefault branches:\n";
         for (auto &&branch : defBranches) {
            ret << " - " << branch << "\n";
         }
      }
   }

   return ret.str();
}
} // namespace cling

#endif // ROOT_TDATAFRAME
