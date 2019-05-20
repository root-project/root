/// \file RPageStorage.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RField.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RPageStorageRoot.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/TLogger.hxx>

#include <TKey.h>

#include <cstdlib>
#include <iostream>
#include <utility>


ROOT::Experimental::Detail::RPageSinkRoot::RPageSinkRoot(std::string_view forestName, RSettings settings)
   : ROOT::Experimental::Detail::RPageSink(forestName)
   , fForestName(forestName)
   , fDirectory(nullptr)
   , fSettings(settings)
   , fPrevClusterNEntries(0)
{
   R__WARNING_HERE("Forest") << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";
}

ROOT::Experimental::Detail::RPageSinkRoot::RPageSinkRoot(std::string_view forestName, std::string_view path)
   : ROOT::Experimental::Detail::RPageSink(forestName)
   , fForestName(forestName)
   , fDirectory(nullptr)
{
   R__WARNING_HERE("Forest") << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";
   TFile *file = TFile::Open(std::string(path).c_str(), "UPDATE");
   fSettings.fFile = file;
   fSettings.fTakeOwnership = true;
}

ROOT::Experimental::Detail::RPageSinkRoot::~RPageSinkRoot()
{
   if (fSettings.fTakeOwnership) {
      fSettings.fFile->Close();
      delete fSettings.fFile;
   }
}

ROOT::Experimental::Detail::RPageStorage::ColumnHandle_t
ROOT::Experimental::Detail::RPageSinkRoot::AddColumn(RColumn* column)
{
   ROOT::Experimental::Internal::RColumnHeader columnHeader;
   columnHeader.fName = column->GetModel().GetName();
   columnHeader.fType = column->GetModel().GetType();
   columnHeader.fIsSorted = column->GetModel().GetIsSorted();
   if (column->GetOffsetColumn() != nullptr) {
      columnHeader.fOffsetColumn = column->GetOffsetColumn()->GetModel().GetName();
   }
   auto columnId = fForestHeader.fColumns.size();
   fForestHeader.fColumns.emplace_back(columnHeader);
   //printf("Added column %s type %d\n", columnHeader.fName.c_str(), (int)columnHeader.fType);
   return ColumnHandle_t(columnId, column);
}


void ROOT::Experimental::Detail::RPageSinkRoot::Create(RNTupleModel *model)
{
   fForestHeader.fPageSize = kPageSize;
   fDirectory = fSettings.fFile->mkdir(fForestName.c_str());

   unsigned int nColumns = 0;
   for (auto& f : *model->GetRootField()) {
      nColumns += f.GetNColumns();
   }
   fPagePool = std::make_unique<RPagePool>(fForestHeader.fPageSize, nColumns);

   for (auto& f : *model->GetRootField()) {
      ROOT::Experimental::Internal::RFieldHeader fieldHeader;
      fieldHeader.fName = f.GetName();
      fieldHeader.fType = f.GetType();
      //printf("Added field %s type [%s]\n", f.GetName().c_str(), f.GetType().c_str());
      if (f.GetParent()) fieldHeader.fParentName = f.GetParent()->GetName();
      fForestHeader.fFields.emplace_back(fieldHeader);

      f.ConnectColumns(this); // issues in turn one or several calls to AddColumn()
   }
   R__ASSERT(nColumns == fForestHeader.fColumns.size());

   fCurrentCluster.fPagesPerColumn.resize(nColumns);
   fForestFooter.fNElementsPerColumn.resize(nColumns, 0);
   fDirectory->WriteObject(&fForestHeader, RMapper::kKeyForestHeader);
}

void ROOT::Experimental::Detail::RPageSinkRoot::CommitPage(ColumnHandle_t columnHandle, const RPage &page)
{
   auto columnId = columnHandle.fId;
   ROOT::Experimental::Internal::RPagePayload pagePayload;
   pagePayload.fSize = page.GetSize();
   pagePayload.fContent = static_cast<unsigned char *>(page.GetBuffer());
   std::string key = std::string(RMapper::kKeyPagePayload) +
      std::to_string(fForestFooter.fNClusters) + RMapper::kKeySeparator +
      std::to_string(columnId) + RMapper::kKeySeparator +
      std::to_string(fCurrentCluster.fPagesPerColumn[columnId].fRangeStarts.size());
   fDirectory->WriteObject(&pagePayload, key.c_str());
   fCurrentCluster.fPagesPerColumn[columnId].fRangeStarts.push_back(page.GetRangeFirst());
   fForestFooter.fNElementsPerColumn[columnId] += page.GetNElements();
}

void ROOT::Experimental::Detail::RPageSinkRoot::CommitCluster(ROOT::Experimental::ForestSize_t nEntries)
{
   fCurrentCluster.fNEntries = nEntries - fPrevClusterNEntries;
   fPrevClusterNEntries = nEntries;
   std::string key = RMapper::kKeyClusterFooter + std::to_string(fForestFooter.fNClusters);
   fDirectory->WriteObject(&fCurrentCluster, key.c_str());
   fForestFooter.fNClusters++;
   fForestFooter.fNEntries = nEntries;

   for (auto& pageInfo : fCurrentCluster.fPagesPerColumn) {
      pageInfo.fRangeStarts.clear();
   }
   fCurrentCluster.fEntryRangeStart = fForestFooter.fNEntries;
}


void ROOT::Experimental::Detail::RPageSinkRoot::CommitDataset()
{
   if (fDirectory)
      fDirectory->WriteObject(&fForestFooter, RMapper::kKeyForestFooter);
}


//------------------------------------------------------------------------------


ROOT::Experimental::Detail::RPageSourceRoot::RPageSourceRoot(std::string_view forestName, RSettings settings)
   : ROOT::Experimental::Detail::RPageSource(forestName)
   , fForestName(forestName)
   , fDirectory(nullptr)
   , fSettings(settings)
{
}

ROOT::Experimental::Detail::RPageSourceRoot::RPageSourceRoot(std::string_view forestName, std::string_view path)
   : ROOT::Experimental::Detail::RPageSource(forestName)
   , fForestName(forestName)
   , fDirectory(nullptr)
{
   TFile *file = TFile::Open(std::string(path).c_str(), "READ");
   fSettings.fFile = file;
   fSettings.fTakeOwnership = true;
}


ROOT::Experimental::Detail::RPageSourceRoot::~RPageSourceRoot()
{
   if (fSettings.fTakeOwnership) {
      fSettings.fFile->Close();
      delete fSettings.fFile;
   }
}


ROOT::Experimental::Detail::RPageStorage::ColumnHandle_t
ROOT::Experimental::Detail::RPageSourceRoot::AddColumn(RColumn* column)
{
   auto& model = column->GetModel();
   auto columnId = fMapper.fColumnName2Id[model.GetName()];
   R__ASSERT(model == *fMapper.fId2ColumnModel[columnId]);
   //printf("Attaching column %s id %d type %d length %lu\n",
   //   column->GetModel().GetName().c_str(), columnId, (int)(column->GetModel().GetType()),
   //   fMapper.fColumnIndex[columnId].fNElements);
   return ColumnHandle_t(columnId, column);
}


void ROOT::Experimental::Detail::RPageSourceRoot::Attach()
{
   fDirectory = fSettings.fFile->GetDirectory(fForestName.c_str());
   auto keyForestHeader = fDirectory->GetKey(RMapper::kKeyForestHeader);
   auto forestHeader = keyForestHeader->ReadObject<ROOT::Experimental::Internal::RForestHeader>();
   //printf("Number of fields %lu, of columns %lu\n", forestHeader->fFields.size(), forestHeader->fColumns.size());

   for (auto &fieldHeader : forestHeader->fFields) {
      if (fieldHeader.fParentName.empty()) {
         fMapper.fRootFields.push_back(RMapper::RFieldDescriptor(fieldHeader.fName, fieldHeader.fType));
      }
   }

   auto nColumns = forestHeader->fColumns.size();
   fPagePool = std::make_unique<RPagePool>(forestHeader->fPageSize, nColumns);
   fMapper.fColumnIndex.resize(nColumns);

   std::int32_t columnId = 0;
   for (auto &columnHeader : forestHeader->fColumns) {
      auto columnModel = std::make_unique<RColumnModel>(
         columnHeader.fName, columnHeader.fType, columnHeader.fIsSorted);
      fMapper.fId2ColumnModel[columnId] = std::move(columnModel);
      fMapper.fColumnName2Id[columnHeader.fName] = columnId;
      columnId++;
   }

   /// Determine column dependencies (offset - pointee relationships)
   for (auto &columnHeader : forestHeader->fColumns) {
      if (columnHeader.fOffsetColumn.empty()) continue;
      fMapper.fColumn2Pointee[fMapper.fColumnName2Id[columnHeader.fOffsetColumn]] =
        fMapper.fColumnName2Id[columnHeader.fName];
   }

   auto keyForestFooter = fDirectory->GetKey(RMapper::kKeyForestFooter);
   auto forestFooter = keyForestFooter->ReadObject<ROOT::Experimental::Internal::RForestFooter>();
   //printf("Number of clusters: %d, entries %ld\n", forestFooter->fNClusters, forestFooter->fNEntries);

   for (std::int32_t iCluster = 0; iCluster < forestFooter->fNClusters; ++iCluster) {
      auto keyClusterFooter = fDirectory->GetKey((RMapper::kKeyClusterFooter + std::to_string(iCluster)).c_str());
      auto clusterFooter = keyClusterFooter->ReadObject<ROOT::Experimental::Internal::RClusterFooter>();
      R__ASSERT(clusterFooter->fPagesPerColumn.size() == nColumns);
      for (unsigned iColumn = 0; iColumn < nColumns; ++iColumn) {
         if (clusterFooter->fPagesPerColumn[iColumn].fRangeStarts.empty())
            continue;
         ForestSize_t selfClusterOffset = clusterFooter->fPagesPerColumn[iColumn].fRangeStarts[0];
         ForestSize_t pointeeClusterOffset = kInvalidForestIndex;
         auto itrPointee = fMapper.fColumn2Pointee.find(iColumn);
         if (itrPointee != fMapper.fColumn2Pointee.end()) {
            //printf("COLUMN %s wants to know pointee offset of column %s\n",
            //  fMapper.fId2ColumnModel[iColumn]->GetName().c_str(),
            //  fMapper.fId2ColumnModel[itrPointee->second]->GetName().c_str());
            /// The pointee might not have any pages in this cluster (e.g. all empty collections)
            if (!clusterFooter->fPagesPerColumn[itrPointee->second].fRangeStarts.empty())
               pointeeClusterOffset = clusterFooter->fPagesPerColumn[itrPointee->second].fRangeStarts[0];
         }
         ForestSize_t pageInCluster = 0;
         for (auto rangeStart : clusterFooter->fPagesPerColumn[iColumn].fRangeStarts) {
            fMapper.fColumnIndex[iColumn].fRangeStarts.push_back(rangeStart);
            fMapper.fColumnIndex[iColumn].fClusterId.push_back(iCluster);
            fMapper.fColumnIndex[iColumn].fPageInCluster.push_back(pageInCluster);
            fMapper.fColumnIndex[iColumn].fSelfClusterOffset.push_back(selfClusterOffset);
            fMapper.fColumnIndex[iColumn].fPointeeClusterOffset.push_back(pointeeClusterOffset);
            pageInCluster++;
         }
      }
      delete clusterFooter;
   }

   for (unsigned iColumn = 0; iColumn < nColumns; ++iColumn) {
      fMapper.fColumnIndex[iColumn].fNElements = forestFooter->fNElementsPerColumn[iColumn];
   }
   fMapper.fNEntries = forestFooter->fNEntries;

   delete forestFooter;
   delete forestHeader;

   // TODO(jblomer): replace RMapper by a forest descriptor
   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetForest(fForestName, RForestVersion());
   fDescriptor = descBuilder.GetDescriptor();
}


std::unique_ptr<ROOT::Experimental::RNTupleModel> ROOT::Experimental::Detail::RPageSourceRoot::GenerateModel()
{
   auto model = std::make_unique<RNTupleModel>();
   for (auto& f : fMapper.fRootFields) {
      auto field = Detail::RFieldBase::Create(f.fFieldName, f.fTypeName);
      model->AddField(std::unique_ptr<Detail::RFieldBase>(field));
   }
   return model;
}

void ROOT::Experimental::Detail::RPageSourceRoot::PopulatePage(
   ColumnHandle_t columnHandle, ForestSize_t index, RPage* page)
{
   auto columnId = columnHandle.fId;
   auto nElems = fMapper.fColumnIndex[columnId].fNElements;
   R__ASSERT(index < nElems);

   ForestSize_t firstInPage = 0;
   ForestSize_t firstOutsidePage = nElems;
   ForestSize_t pageIdx = 0;

   std::size_t iLower = 0;
   std::size_t iUpper = fMapper.fColumnIndex[columnId].fRangeStarts.size() - 1;
   R__ASSERT(iLower <= iUpper);
   unsigned iLast = iUpper;
   while (iLower <= iUpper) {
      std::size_t iPivot = (iLower + iUpper) / 2;
      ForestSize_t pivot = fMapper.fColumnIndex[columnId].fRangeStarts[iPivot];
      if (pivot > index) {
         iUpper = iPivot - 1;
      } else {
         auto next = nElems;
         if (iPivot < iLast) next = fMapper.fColumnIndex[columnId].fRangeStarts[iPivot + 1];
         if ((pivot == index) || (next > index)) {
            firstOutsidePage = next;
            firstInPage = pivot;
            pageIdx = iPivot;
            break;
         } else {
            iLower = iPivot + 1;
         }
      }
   }

   auto elemsInPage = firstOutsidePage - firstInPage;
   void* buf = page->TryGrow(elemsInPage);
   R__ASSERT(buf != nullptr);

   auto clusterId = fMapper.fColumnIndex[columnId].fClusterId[pageIdx];
   auto pageInCluster = fMapper.fColumnIndex[columnId].fPageInCluster[pageIdx];
   auto selfOffset = fMapper.fColumnIndex[columnId].fSelfClusterOffset[pageIdx];
   auto pointeeOffset = fMapper.fColumnIndex[columnId].fPointeeClusterOffset[pageIdx];
   page->SetWindow(firstInPage, RPage::RClusterInfo(clusterId, selfOffset, pointeeOffset));

   //printf("Populating page %lu/%lu [%lu] for column %d starting at %lu\n", clusterId, pageInCluster, pageIdx, columnId, firstInPage);

   std::string keyName = std::string(RMapper::kKeyPagePayload) +
      std::to_string(clusterId) + RMapper::kKeySeparator +
      std::to_string(columnId) + RMapper::kKeySeparator +
      std::to_string(pageInCluster);
   auto pageKey = fDirectory->GetKey(keyName.c_str());
   auto pagePayload = pageKey->ReadObject<ROOT::Experimental::Internal::RPagePayload>();
   R__ASSERT(static_cast<std::size_t>(pagePayload->fSize) == page->GetSize());
   memcpy(page->GetBuffer(), pagePayload->fContent, pagePayload->fSize);

   free(pagePayload->fContent);
   free(pagePayload);
}

ROOT::Experimental::ForestSize_t ROOT::Experimental::Detail::RPageSourceRoot::GetNEntries()
{
   return fMapper.fNEntries;
}

ROOT::Experimental::ForestSize_t ROOT::Experimental::Detail::RPageSourceRoot::GetNElements(ColumnHandle_t columnHandle)
{
   return fMapper.fColumnIndex[columnHandle.fId].fNElements;
}

ROOT::Experimental::ColumnId_t ROOT::Experimental::Detail::RPageSourceRoot::GetColumnId(ColumnHandle_t columnHandle)
{
   // TODO(jblomer) distinguish trees
   return columnHandle.fId;
}
