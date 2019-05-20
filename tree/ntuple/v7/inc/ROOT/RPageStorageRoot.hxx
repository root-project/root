/// \file ROOT/RPageStorage.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorageRoot
#define ROOT7_RPageStorageRoot

#include <ROOT/RPageStorage.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <TDirectory.h>
#include <TFile.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

namespace Internal {

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
   std::string fOffsetColumn;
};

struct RForestHeader {
   std::int32_t fVersion = 0;
   std::string fModelUuid;
   std::int32_t fPageSize = 0;
   std::vector<RFieldHeader> fFields;
   std::vector<RColumnHeader> fColumns;
};

struct RForestFooter {
   std::int32_t fVersion = 0;
   std::int32_t fNClusters = 0;
   ForestSize_t fNEntries = 0;
   std::vector<ForestSize_t> fNElementsPerColumn;
};

struct RPageInfo {
   std::vector<ForestSize_t> fRangeStarts;
};

struct RClusterFooter {
   std::int32_t fVersion = 0;
   ForestSize_t fEntryRangeStart = 0;
   ForestSize_t fNEntries = 0;
   std::vector<RPageInfo> fPagesPerColumn;
};

struct RPagePayload {
   std::int32_t fVersion = 0;
   int fSize = 0;
   unsigned char* fContent = nullptr; //[fSize]
};

} // namespace Internal


namespace Detail {

class RPagePool;

/**
 * Maps the Forest meta-data to and from TFile
 */
class RMapper {
public:
   static constexpr const char* kKeySeparator = "_";
   static constexpr const char* kKeyForestHeader = "RFH";
   static constexpr const char* kKeyForestFooter = "RFF";
   static constexpr const char* kKeyClusterFooter = "RFCF";
   static constexpr const char* kKeyPagePayload = "RFP";

   struct RColumnIndex {
      ForestSize_t fNElements = 0;
      std::vector<ForestSize_t> fRangeStarts;
      std::vector<ForestSize_t> fClusterId;
      std::vector<ForestSize_t> fPageInCluster;
      std::vector<ForestSize_t> fSelfClusterOffset;
      std::vector<ForestSize_t> fPointeeClusterOffset;
   };

   struct RFieldDescriptor {
      RFieldDescriptor(const std::string &f, const std::string &t) : fFieldName(f), fTypeName(t) {}
      std::string fFieldName;
      std::string fTypeName;
   };

   ForestSize_t fNEntries = 0;
   std::unordered_map<std::int32_t, std::unique_ptr<RColumnModel>> fId2ColumnModel;
   std::unordered_map<std::string, std::int32_t> fColumnName2Id;
   std::unordered_map<std::int32_t, std::int32_t> fColumn2Pointee;
   std::vector<RColumnIndex> fColumnIndex;
   std::vector<RFieldDescriptor> fRootFields;
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
   static constexpr std::size_t kPageSize = 32000;

   std::string fForestName;
   /// Currently, a forest is stored as a directory in a TFile
   TDirectory *fDirectory;
   RSettings fSettings;
   /// Updated on CommitPage and written and reset on CommitCluster
   ROOT::Experimental::Internal::RClusterFooter fCurrentCluster;
   ROOT::Experimental::Internal::RForestHeader fForestHeader;
   ROOT::Experimental::Internal::RForestFooter fForestFooter;

   RMapper fMapper;
   ForestSize_t fPrevClusterNEntries;

public:
   RPageSinkRoot(std::string_view forestName, RSettings settings);
   RPageSinkRoot(std::string_view forestName, std::string_view path);
   virtual ~RPageSinkRoot();

   ColumnHandle_t AddColumn(RColumn* column) final;
   void Create(RForestModel* model) final;
   void CommitPage(ColumnHandle_t columnHandle, const RPage &page) final;
   void CommitCluster(ForestSize_t nEntries) final;
   void CommitDataset() final;
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
      bool fTakeOwnership = false;
   };

private:
   std::string fForestName;
   /// Currently, a forest is stored as a directory in a TFile
   TDirectory *fDirectory;
   RSettings fSettings;

   RMapper fMapper;
   RForestDescriptor fDescriptor;

public:
   RPageSourceRoot(std::string_view forestName, RSettings settings);
   RPageSourceRoot(std::string_view forestName, std::string_view path);
   virtual ~RPageSourceRoot();

   ColumnHandle_t AddColumn(RColumn* column) final;
   void Attach() final;
   std::unique_ptr<ROOT::Experimental::RForestModel> GenerateModel() final;
   void PopulatePage(ColumnHandle_t columnHandle, ForestSize_t index, RPage* page) final;
   ForestSize_t GetNEntries() final;
   ForestSize_t GetNElements(ColumnHandle_t columnHandle) final;
   ColumnId_t GetColumnId(ColumnHandle_t columnHandle) final;
   const RForestDescriptor& GetDescriptor() const final { return fDescriptor; }
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
