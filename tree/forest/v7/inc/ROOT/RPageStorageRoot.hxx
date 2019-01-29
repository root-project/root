/// \file ROOT/RPageStorage.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorageRoot
#define ROOT7_RPageStorageRoot

#include <ROOT/RPageStorage.hxx>
#include <ROOT/RColumnModel.hxx>

#include <TDirectory.h>
#include <TFile.h>

#include <string>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

namespace Internal {

struct RForestHeader {
   std::int32_t fVersion = 0;
   std::int32_t fNFields = 0;
   std::int32_t fNColumns = 0;
   std::string fModelUuid;
};

struct RFieldHeader {
   std::int32_t fVersion = 0;
   std::string fName;
   std::string fType;
   std::string fParentName;
};

struct RColumnHeader {
   std::int32_t fVersion = 0;
   std::string fName;
   EColumnType fType;
   bool fIsSorted;
};

struct RClusterHeader {
   std::int32_t fVersion = 0;
};

} // namespace Internal


namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkRoot
\ingroup Forest
\brief Abstract interface to write data into a tree

The page sink takes the list of columns and afterwards a series of page commits and cluster commits.
The user is responsible to commit clusters at consistent point, i.e. when all pages corresponding to data
up to the given entry number are committed.
*/
// clang-format on
class RPageSinkRoot : public RPageSink {
public:
   struct RSettings {
      TFile *fFile = nullptr;
   };

private:
   static constexpr const char* kKeyForestHeader = "RFH";
   static constexpr const char* kKeyFieldHeader = "RFFH";
   static constexpr const char* kKeyColumnHeader = "RFCH";

   std::string fForestName;
   /// Currently, a forest is stored as a directory in a TFile
   TDirectory *fDirectory;
   RSettings fSettings;

   std::unordered_map<RColumn*, int> fColumn2Id;

public:
   RPageSinkRoot(std::string_view forestName, RSettings settings);
   virtual ~RPageSinkRoot();

   /// TODO(jblomer): keep abtract and let derived classed define
   void AddColumn(RColumn* /*column*/) final;

   /// Physically creates the storage container to hold the tree (e.g., a directory in a TFile or a S3 bucket)
   void Create(RTreeModel* model) final;
   /// Write a page to the storage. The column attached to the page must have been added before.
   void CommitPage(RPage* page) final;
   /// Finalize the current cluster and create a new one for the following data.
   void CommitCluster(TreeIndex_t nEntries) final;
   /// Finalize the current cluster and the entrire data set.
   void CommitDataset(TreeIndex_t nEntries) final;
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
