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

#include <memory>
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

/**
 * Maps the Forest meta-data to and from TFile
 */
class RMapper {
public:
   static constexpr const char* kKeyForestHeader = "RFH";
   static constexpr const char* kKeyFieldHeader = "RFFH";
   static constexpr const char* kKeyColumnHeader = "RFCH";

   std::unordered_map<RColumn*, std::int32_t> fColumn2Id;
   std::unordered_map<std::int32_t, std::unique_ptr<RColumnModel>> fId2ColumnModel;
   std::unordered_map<std::string, std::int32_t> fColumnName2Id;
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSinkRoot
\ingroup Forest
\brief Storage provider that write Forest pages into a ROOT TFile
*/
// clang-format on
class RPageSinkRoot : public RPageSink {
public:
   struct RSettings {
      TFile *fFile = nullptr;
      bool fTakeOwnership = false;
   };

private:
   std::string fForestName;
   /// Currently, a forest is stored as a directory in a TFile
   TDirectory *fDirectory;
   RSettings fSettings;

   RMapper fMapper;

public:
   RPageSinkRoot(std::string_view forestName, RSettings settings);
   virtual ~RPageSinkRoot();

   void AddColumn(RColumn* column) final;
   void Create(RTreeModel* model) final;
   void CommitPage(RPage* page) final;
   void CommitCluster(TreeIndex_t nEntries) final;
   void CommitDataset(TreeIndex_t nEntries) final;
};


// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSourceRoot
\ingroup Forest
\brief Storage provider that reads Forest pages from a ROOT TFile
*/
// clang-format on
class RPageSourceRoot : public RPageSource {
public:
   struct RSettings {
      TFile *fFile = nullptr;
   };

private:
   std::string fForestName;
   /// Currently, a forest is stored as a directory in a TFile
   TDirectory *fDirectory;
   RSettings fSettings;

   ROOT::Experimental::Internal::RForestHeader* fForestHeader;
   RMapper fMapper;

public:
   RPageSourceRoot(std::string_view forestName, RSettings settings);
   virtual ~RPageSourceRoot();

   void AddColumn(RColumn* /*column*/) final;
   void Attach() final;
   std::unique_ptr<ROOT::Experimental::RTreeModel> GenerateModel() final;
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
