/// \file RDrawStorage.cxx
/// \ingroup NTupleDraw ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-11-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDrawStorage.hxx>

#include <ROOT/RColumnModel.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <Rtypes.h>

#include <Buttons.h>
#include <TBox.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TLine.h>
#include <TPad.h>
#include <TStyle.h>
#include <TText.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

std::vector<ROOT::Experimental::Detail::RDrawStorage> ROOT::Experimental::Detail::RDrawStorage::fgDrawStorageVec;

// ------------------------------ RMetaDataBox ------------------------------

ROOT::Experimental::Detail::RMetaDataBox::RMetaDataBox(double x1, double y1, double x2, double y2,
                                                       std::string description, std::uint32_t nBytes,
                                                       RDrawStorage *parent, std::int32_t color)
   : TBox(x1, y1, x2, y2), fDescription{description}, fNBytesInStorage{nBytes}, fParent{parent}
{
   SetFillColor(color);
}

void ROOT::Experimental::Detail::RMetaDataBox::Dump() const
{
   std::cout << " ==> Dumping Page information:\n\n";
   std::cout << "Description:            \t\t" << fDescription << std::endl;
   std::cout << "Size:                   \t\t" << fNBytesInStorage << " bytes" << std::endl;
}

void ROOT::Experimental::Detail::RMetaDataBox::Inspect() const
{
   static std::int32_t index = 0;
   // The canvases need to have unique names, or else there will be an error saying that not all were found when trying
   // to delete them when quitting the program.
   std::string uniqueCanvasName{"MetaDataDetails" + std::to_string(++index)};

   fParent->fCanvasPtrs.emplace_back(new TCanvas(uniqueCanvasName.c_str(), "Page Details", 500, 300));
   TLatex latex;

   // Draw Title
   latex.SetTextAlign(12);
   latex.SetTextSize(0.08);
   latex.DrawLatex(0.01, 0.96, fDescription.c_str());

   // Write Details
   latex.SetTextSize(0.06);
   std::string sizeString = "Size:" + std::string(30, ' ') + std::to_string(fNBytesInStorage) + " bytes";
   latex.DrawLatex(0.01, 0.85, sizeString.c_str());
   fParent->fCanvasPtrs.back()->Update();
}

ClassImp(ROOT::Experimental::Detail::RMetaDataBox)

   // ------------------------------ RPageBox ----------------------------------

   void ROOT::Experimental::Detail::RPageBox::Dump() const
{
   std::cout << " ==> Dumping Page information:\n\n";
   std::cout << "Page Id:                \t\t" << fPageBoxId << " / " << fParent->GetPageBoxSize() << std::endl;
   std::cout << "Cluster Id:             \t\t" << fClusterId << " / " << fParent->GetNClusters() - 1 << std::endl;
   std::cout << "Field Id:               \t\t" << fFieldId << " / " << fParent->GetNFields() - 1 << std::endl;
   std::cout << "FieldName:              \t\t" << fFieldName << std::endl;
   std::cout << "FieldType:              \t\t" << fFieldType << std::endl;
   std::cout << "Column Id:              \t\t" << fColumnId << " / " << fParent->GetNColumns() - 1 << std::endl;
   std::cout << "ColumnType:             \t\t" << fColumnType << std::endl;
   std::cout << "NElements:              \t\t" << fNElements << std::endl;
   std::cout << "Element Size On Disk:   \t\t" << fElementSizeOnDisk << " bits" << std::endl;
   std::cout << "Element Size On Storage:\t\t" << 8 * fLocator.fBytesOnStorage / fNElements << " bits" << std::endl;
   std::cout << "Page Size On Disk:      \t\t" << fNElements * fElementSizeOnDisk / 8 << " bytes" << std::endl;
   std::cout << "Page Size On Storage:   \t\t" << fLocator.fBytesOnStorage << " bytes" << std::endl;
   std::cout << "Global Page Range:      \t\t" << fGlobalRangeStart << " - " << fGlobalRangeStart + fNElements - 1
             << std::endl;
   std::cout << "Cluster Page Range:     \t\t" << fClusterRangeStart << " - " << fClusterRangeStart + fNElements - 1
             << std::endl;
   std::size_t totalNumBytes = fParent->GetTotalNumBytes();
   std::size_t scalingFactorOfAxis = fParent->GetScalingFactorOfAxis();
   std::cout.setf(std::ios::fixed);
   std::cout << "Location in Storage:    \t\t" << static_cast<std::size_t>(GetX1() * scalingFactorOfAxis) << " / "
             << totalNumBytes << " bytes" << std::endl;
   std::cout.unsetf(std::ios::fixed);
}

void ROOT::Experimental::Detail::RPageBox::Inspect() const
{
   static std::int32_t index = 0;
   // The canvases need to have unique names, or else there will be an error saying that not all were found when trying
   // to delete them when quitting the program.
   std::string uniqueCanvasName{"PageDetails" + std::to_string(++index)};
   fParent->fCanvasPtrs.emplace_back(new TCanvas(uniqueCanvasName.c_str(), "Page Details", 500, 300));

   TLatex latex;
   // Draw Title
   latex.SetTextAlign(12);
   latex.SetTextSize(0.08);
   std::string pageNumbering =
      "Page No. " + std::to_string(fPageBoxId) + " / " + std::to_string(fParent->GetPageBoxSize());
   latex.DrawLatex(0.01, 0.96, pageNumbering.c_str());

   // Write details about page
   latex.SetTextSize(0.06);
   std::string clusterIdString = "Cluster Id:" + std::string(37, ' ') + std::to_string(fClusterId) + " / " +
                                 std::to_string(fParent->GetNClusters() - 1);
   latex.DrawLatex(0.01, 0.85, clusterIdString.c_str());
   std::string fieldIdString =
      "Field Id:" + std::string(41, ' ') + std::to_string(fFieldId) + " / " + std::to_string(fParent->GetNFields() - 1);
   latex.DrawLatex(0.01, 0.80, fieldIdString.c_str());
   std::string fieldName = "FieldName:" + std::string(35, ' ') + fFieldName;
   latex.DrawLatex(0.01, 0.75, fieldName.c_str());
   std::string fieldType = "FieldType:" + std::string(37, ' ') + fFieldType;
   latex.DrawLatex(0.01, 0.70, fieldType.c_str());
   std::string columnIdString = "Column Id:" + std::string(36, ' ') + std::to_string(fColumnId) + " / " +
                                std::to_string(fParent->GetNColumns() - 1);
   latex.DrawLatex(0.01, 0.65, columnIdString.c_str());
   std::string columnTypeString = "ColumnType:" + std::string(32, ' ') + fColumnType;
   latex.DrawLatex(0.01, 0.60, columnTypeString.c_str());
   std::string nElements = "NElements:" + std::string(35, ' ') + std::to_string(fNElements);
   latex.DrawLatex(0.01, 0.55, nElements.c_str());
   std::string elementSizeOnDisk =
      "Element Size On Disk:" + std::string(17, ' ') + std::to_string(fElementSizeOnDisk) + " bits";
   latex.DrawLatex(0.01, 0.50, elementSizeOnDisk.c_str());
   std::string elementSizeOnStorage = "Element Size On Storage:" + std::string(11, ' ') +
                                      std::to_string(8 * fLocator.fBytesOnStorage / fNElements) + " bits";
   latex.DrawLatex(0.01, 0.45, elementSizeOnStorage.c_str());
   std::string pageSize =
      "Page Size On Disk:" + std::string(22, ' ') + std::to_string(fNElements * fElementSizeOnDisk / 8) + " bytes";
   latex.DrawLatex(0.01, 0.40, pageSize.c_str());
   std::string pageSizeStorage =
      "Page Size On Storage:" + std::string(16, ' ') + std::to_string(fLocator.fBytesOnStorage) + " bytes";
   latex.DrawLatex(0.01, 0.35, pageSizeStorage.c_str());
   std::string globalRange = "Global Page Range:" + std::string(21, ' ') + std::to_string(fGlobalRangeStart) + " - " +
                             std::to_string(fGlobalRangeStart + fNElements - 1);
   latex.DrawLatex(0.01, 0.30, globalRange.c_str());
   std::string clusterRange = "Cluster Page Range:" + std::string(20, ' ') + std::to_string(fClusterRangeStart) +
                              " - " + std::to_string(fClusterRangeStart + fNElements - 1);
   latex.DrawLatex(0.01, 0.25, clusterRange.c_str());
   std::size_t totalNumBytes = fParent->GetTotalNumBytes();
   std::size_t scalingFactorOfAxis = fParent->GetScalingFactorOfAxis();
   std::string locationString = "Location in Storage:" + std::string(20, ' ') +
                                std::to_string(static_cast<std::size_t>(GetX1() * scalingFactorOfAxis)) + " / " +
                                std::to_string(totalNumBytes) + " bytes";
   latex.DrawLatex(0.01, 0.20, locationString.c_str());

   fParent->fCanvasPtrs.back()->Update();
}

ClassImp(ROOT::Experimental::Detail::RPageBox)

   // ------------------------------ RDrawStorage ------------------------------

   ROOT::Experimental::Detail::RPageBox::RPageBox(double x1, double y1, double x2, double y2, std::string fieldName,
                                                  std::string fieldType, DescriptorId_t fieldId,
                                                  DescriptorId_t columnId, DescriptorId_t clusterId,
                                                  EColumnType columnType, ClusterSize_t::ValueType nElements,
                                                  NTupleSize_t globalRangeStart, NTupleSize_t clusterRangeStart,
                                                  RClusterDescriptor::RLocator locator, RDrawStorage *parent,
                                                  std::size_t pageBoxId)
   : TBox(x1, y1, x2, y2), fFieldName{fieldName}, fFieldType{fieldType}, fFieldId{fieldId}, fColumnId{columnId},
     fClusterId{clusterId}, fNElements{nElements}, fGlobalRangeStart{globalRangeStart},
     fClusterRangeStart{clusterRangeStart}, fLocator{locator}, fParent{parent}, fPageBoxId{pageBoxId}
{
   switch (columnType) {
   case EColumnType::kIndex:
      fColumnType = "Index";
      fElementSizeOnDisk = sizeof(ClusterSize_t) * 8;
      break;
   case EColumnType::kSwitch:
      fColumnType = "Switch";
      fElementSizeOnDisk = sizeof(ROOT::Experimental::RColumnSwitch) * 8;
      break;
   case EColumnType::kByte:
      fColumnType = "Byte";
      fElementSizeOnDisk = sizeof(char) * 8;
      break;
   case EColumnType::kBit:
      fColumnType = "Bit";
      fElementSizeOnDisk = sizeof(bool) * 8;
      break;
   case EColumnType::kReal64:
      fColumnType = "Real64";
      fElementSizeOnDisk = sizeof(double) * 8;
      break;
   case EColumnType::kReal32:
      fColumnType = "Real32";
      fElementSizeOnDisk = sizeof(float) * 8;
      break;
   // Uncomment after implementing custom-sized float-packing.
   /*case EColumnType::kReal24:
      fColumnType = "Real24";
      fElementSizeOnDisk = 24;
      break;
   case EColumnType::kCustomDouble:
      fColumnType = "CustomDouble";
      fElementSizeOnDisk = sizeof(double)*8;
      break;
   case EColumnType::kCustomFloat:
      fColumnType = "CustomFloat";
      fElementSizeOnDisk = sizeof(float)*8;
      break;*/
   case EColumnType::kReal16:
      fColumnType = "Real16";
      fElementSizeOnDisk = 16;
      break;
   case EColumnType::kReal8:
      fColumnType = "Real8";
      fElementSizeOnDisk = 8;
      break;
   case EColumnType::kInt64:
      fColumnType = "Int64";
      fElementSizeOnDisk = 64;
      break;
   case EColumnType::kInt32:
      fColumnType = "Int32";
      fElementSizeOnDisk = 32;
      break;
   case EColumnType::kInt16:
      fColumnType = "Int16";
      fElementSizeOnDisk = 16;
      break;
   case EColumnType::kUnknown:
      fColumnType = "kUnknown";
      fElementSizeOnDisk = -1;
      break;
   default: assert(false);
   }
}

ROOT::Experimental::Detail::RDrawStorage::RDrawStorage(ROOT::Experimental::RNTupleReader *reader)
{
   // RDrawStorage::Draw() should work without the descriptor because its lifetime is not coupled to this object. The
   // job of this constructor is to get all information necessary for drawing from the descriptor, so that it isn't
   // required later anymore.
   auto &desc = reader->GetDescriptor();

   /* Procedure:
    * 1. Check for special cases like empty ntuple
    * 2. Prepare Title and TLegend
    * 3. Create all boxes and colour them
    * 4. Sort RPageBoxes by page order and set Page Ids
    * 5. Create CumulativeNBytes to later to set x1 and x2 values for the boxes
    * 6. Prepare StorageSizeAxis
    * 7. Set x1 and x2 values for all boxes
    * 8. Include box entries in the legend
    * 9. Prepare ClusterAxis
    */

   // 1. Check for special cases like empty ntuple
   fNFields = desc.GetNFields();
   fNEntries = desc.GetNEntries();
   if (fNFields <= 1 || fNEntries <= 0)
      return;

   // 2. Prepare Title and TLegend
   fNTupleName = desc.GetName();
   std::string title = "Storage layout of " + fNTupleName;
   fTexts.emplace_back(std::make_unique<TText>(.5, .94, title.c_str()));
   fTexts.back()->SetTextAlign(22);
   fTexts.back()->SetTextSize(0.08);

   fLegend = std::make_unique<TLegend>(0.05, 0.05, .95, .55);
   std::int32_t nColumnsInLegend = 2;
   if (fNFields > 150) {
      nColumnsInLegend = 10;
   } else if (fNFields > 120) {
      nColumnsInLegend = 9;
   } else if (fNFields > 100) {
      nColumnsInLegend = 8;
   } else if (fNFields > 75) {
      nColumnsInLegend = 7;
   } else if (fNFields > 33) {
      nColumnsInLegend = 6;
   } else if (fNFields > 26) {
      nColumnsInLegend = 5;
   } else if (fNFields > 19) {
      nColumnsInLegend = 4;
   } else if (fNFields > 4) {
      nColumnsInLegend = 3;
   }
   fLegend->SetNColumns(nColumnsInLegend);

   // 3. Create all boxes and colour them
   constexpr double boxY1 = 0;
   constexpr double boxY2 = 1;
   fHeaderBox =
      std::make_unique<RMetaDataBox>(0.0, boxY1, 0.0, boxY2, "Header", desc.SerializeHeader(nullptr), this, kGray);
   fFooterBox =
      std::make_unique<RMetaDataBox>(0.0, boxY1, 0.0, boxY2, "Footer", desc.SerializeFooter(nullptr), this, kGray);
   fNColumns = desc.GetNColumns();
   fNClusters = desc.GetNClusters();
   for (std::size_t i = 0; i < fNClusters; ++i) {
      for (std::size_t j = 0; j < fNColumns; ++j) {
         ClusterSize_t::ValueType localIndex{0};
         auto &pageRange = desc.GetClusterDescriptor(i).GetPageRange(j);
         for (std::size_t k = 0; k < pageRange.fPageInfos.size(); ++k) {
            // Only use the descriptor to obtain information about a page
            DescriptorId_t fieldId = desc.GetColumnDescriptor(j).GetFieldId();
            std::string fieldName = desc.GetFieldDescriptor(fieldId).GetFieldName();
            std::string fieldType = desc.GetFieldDescriptor(fieldId).GetTypeName();
            EColumnType columnType = desc.GetColumnDescriptor(j).GetModel().GetType();
            ClusterSize_t::ValueType nElements = pageRange.fPageInfos.at(k).fNElements;
            auto clusterRangeFirst = localIndex;
            auto globalRangeFirst = localIndex + desc.GetClusterDescriptor(i).GetColumnRange(j).fFirstElementIndex;

            fPageBoxes.emplace_back(std::make_unique<RPageBox>(
               0, boxY1, 0, boxY2, fieldName, fieldType, fieldId, j, i, columnType, nElements, globalRangeFirst,
               clusterRangeFirst, pageRange.fPageInfos.at(k).fLocator, this));
            fPageBoxes.back()->SetFillColor(GetColourFromFieldId(fieldId));

            localIndex += nElements;
         }
      }
   }

   // 4. Sort RPageBoxes by page order and set Page Ids
   std::sort(fPageBoxes.begin(), fPageBoxes.end(),
             [](const std::unique_ptr<RPageBox> &a, const std::unique_ptr<RPageBox> &b) -> bool {
                if (a->GetClusterId() != b->GetClusterId())
                   return a->GetClusterId() < b->GetClusterId();
                return a->GetLocator().fPosition < b->GetLocator().fPosition;
             });
   SetPageIds();

   // 5. Create CumulativeNBytes to later to set x1 and x2 values for the boxes
   // Size is +2, because +1 for header and +1 for footer
   std::vector<std::uint64_t> cumulativeNBytes(fPageBoxes.size() + 2);
   cumulativeNBytes.at(0) = fHeaderBox->GetNBytesInStorage();
   for (std::size_t i = 1; i < cumulativeNBytes.size() - 1; ++i) {
      cumulativeNBytes.at(i) = cumulativeNBytes.at(i - 1) + fPageBoxes.at(i - 1)->GetLocator().fBytesOnStorage;
   }
   cumulativeNBytes.back() = fFooterBox->GetNBytesInStorage() + cumulativeNBytes.at(cumulativeNBytes.size() - 2);

   // 6. Prepare StorageSizeAxis
   fAxisTitle = "#splitline{Data}{#splitline{Size}{#splitlie{in}{Bytes}}}";
   fTotalNumBytes = cumulativeNBytes.back();
   fScalingFactorOfAxis = 1;
   if (fTotalNumBytes > 1024 * 1024 * 1024) {
      fScalingFactorOfAxis = 1024 * 1024 * 1024;
      fAxisTitle = "#splitline{Data}{#splitline{Size}{in GB}}";
   } else if (fTotalNumBytes > 1024 * 1024) {
      fScalingFactorOfAxis = 1024 * 1024;
      fAxisTitle = "#splitline{Data}{#splitline{Size}{in MB}}";
   } else if (fTotalNumBytes > 1024) {
      fScalingFactorOfAxis = 1024;
      fAxisTitle = "#splitline{Data}{#splitline{Size}{in KB}}";
   }

   // 7. Set x1 and x2 values for all boxes
   fHeaderBox->SetX1(0);
   fHeaderBox->SetX2((double)cumulativeNBytes.at(0) / fScalingFactorOfAxis);
   for (std::size_t i = 0; i < fPageBoxes.size(); ++i) {
      fPageBoxes.at(i)->SetX1((double)cumulativeNBytes.at(i) / fScalingFactorOfAxis);
      fPageBoxes.at(i)->SetX2((double)cumulativeNBytes.at(i + 1) / fScalingFactorOfAxis);
   }
   fFooterBox->SetX1((double)cumulativeNBytes.at(cumulativeNBytes.size() - 2) / fScalingFactorOfAxis);
   fFooterBox->SetX2((double)cumulativeNBytes.back() / fScalingFactorOfAxis);

   // 8. Include box entries in the legend
   fLegend->AddEntry(fHeaderBox.get(), "Header", "f");
   // start at 1 to skip rootField
   for (std::size_t i = 1; i < fNFields; ++i) {
      // for each field find the first PageBox which represents that field and add it to the legend
      auto vecIt = std::find_if(fPageBoxes.begin(), fPageBoxes.end(),
                                [i](const std::unique_ptr<RPageBox> &a) -> bool { return a->GetFieldId() == i; });
      if (vecIt != fPageBoxes.end()) {
         fLegend->AddEntry((*vecIt).get(), desc.GetFieldDescriptor(i).GetFieldName().c_str(), "f");
      }
   }
   fLegend->AddEntry(fFooterBox.get(), "Footer", "f");

   // 9. Prepare ClusterAxis
   double distanceBetweenLines = 0.001 * (fTotalNumBytes / fScalingFactorOfAxis);
   std::size_t start = cumulativeNBytes.at(0);
   std::size_t end{0};
   for (std::size_t i = 0; i < fNClusters; ++i) {
      auto &cluster = desc.GetClusterDescriptor(i);
      std::size_t nBytes = cluster.GetLocator().fBytesOnStorage;
      // For some data formats (e.g. root-Files) this value is equal to 0. In that case get nBytes manually from all
      // columns.
      if (nBytes == 0) {
         for (std::size_t j = 0; j < fNColumns; ++j) {
            for (std::size_t k = 0; k < cluster.GetPageRange(j).fPageInfos.size(); ++k) {
               nBytes += cluster.GetPageRange(j).fPageInfos.at(k).fLocator.fBytesOnStorage;
            }
         }
      }
      end = start + nBytes;
      double x1 = (double)start / fScalingFactorOfAxis + distanceBetweenLines / 2;
      double x2 = (double)end / fScalingFactorOfAxis - distanceBetweenLines / 2;
      fLines.emplace_back(std::make_unique<TLine>(x1, 1.05, x2, 1.05));
      fLines.back()->SetLineWidth(3);
      fTexts.emplace_back(std::make_unique<TText>((x1 + x2) / 2, 1.2, std::to_string(i).c_str()));
      fTexts.back()->SetTextAlign(22);
      fTexts.back()->SetTextSize(0.15);
      start = end;
   }
   fLegend->AddEntry(fLines.back().get(), "Cluster Id", "l");
}

void ROOT::Experimental::Detail::RDrawStorage::Draw()
{
   /* Procedure:
    * 1. Check for special cases like empty ntuple
    * 2. Create a new canvas
    * 3. Create a TPad in the canvas so that when zooming only the boxes and axis get zoomed
    * 4. Draw an empty histogram without a y-axis for zooming
    * 5. Draw all boxes and add possibility to click on RPageBox to obtain information about a page
    * 6. Draw clusterAxis
    * 7. Return to canvas, draw title, legend and description of x-axis
    */

   // 1. Check for special cases like empty ntuple
   if (fNFields <= 1) {
      std::cout << "The ntuple has no fields. No storage layout was drawn." << std::endl;
      return;
   }
   if (fNEntries <= 0) {
      std::cout << "The ntuple has no entries. No storage layout was drawn." << std::endl;
      return;
   }

   // 2. Create a new canvas
   static std::int32_t uniqueId = 0;
   // Trying to delete multiple canvases with the same name leads to an error or when two canvases have the same name, only 1 may get deleted, causing a memory leak.
   std::string uniqueCanvasName = "RDrawStorage" + std::to_string(++uniqueId);
   fCanvasPtrs.emplace_back(new TCanvas(uniqueCanvasName.c_str(), fNTupleName.c_str(), 1000, 300));

   // 3. Create a TPad in the canvas so that when zooming only the boxes and axis get zoomed
   constexpr double marginlength = 0.03;
   std::string uniquePadName = "RDrawStoragePad" + std::to_string(uniqueId);
   fPads.emplace_back(std::make_unique<TPad>(uniquePadName.c_str(), "", marginlength, 0.55, 1 - marginlength, 0.87));
   fPads.back()->SetTopMargin(0.2);
   fPads.back()->SetBottomMargin(0.2);
   fPads.back()->SetLeftMargin(0.01);
   fPads.back()->SetRightMargin(0.01);
   fPads.back()->Draw();
   fPads.back()->cd();

   // 4. Draw an empty histogram without a y-axis for zooming
   std::string uniqueTH1FName = "RDrawStorageTH1F" + std::to_string(uniqueId);
   fTH1Fs.emplace_back(std::make_unique<TH1F>(uniqueTH1FName.c_str(), "", 500, 0, (double)fTotalNumBytes / fScalingFactorOfAxis));
   fTH1Fs.back()->SetMaximum(1);
   fTH1Fs.back()->SetMinimum(0);
   fTH1Fs.back()->GetYaxis()->SetTickLength(0);
   fTH1Fs.back()->GetYaxis()->SetLabelSize(0);
   fTH1Fs.back()->GetXaxis()->SetLabelSize(0.18);
   fTH1Fs.back()->SetStats(0);
   fTH1Fs.back()->DrawCopy();

   // 5. Draw all boxes and add possibility to click on RPageBox to obtain information about a page
   fHeaderBox->Draw();
   for (const auto &b : fPageBoxes) {
      b->Draw();
   }
   fFooterBox->Draw();
   fPads.back()->AddExec("ShowPageDetails", "ROOT::Experimental::Detail::RDrawStorage::RPageBoxClicked()");

   // 6. Draw clusterAxis
   // fTexts.at(0) points to Title so skip
   for (std::size_t i = 1; i < fTexts.size(); ++i) {
      fTexts.at(i)->Draw();
   }
   for (const auto &l : fLines) {
      l->Draw();
   }
   fPads.back()->Update();

   // 7. Return to canvas, draw title, legend and description of x-axis
   fCanvasPtrs.back()->cd();
   fTexts.at(0)->Draw(); // Title
   fLegend->Draw();
   TLatex latex;
   latex.SetTextSize(0.04);
   latex.DrawLatex(0.955, 0.5, fAxisTitle.c_str());

   fCanvasPtrs.back()->Update();
}

void ROOT::Experimental::Detail::RDrawStorage::SetPageIds()
{
   for (std::size_t i = 0; i < fPageBoxes.size(); ++i) {
      fPageBoxes.at(i)->SetPageId(i + 1);
   }
}

void ROOT::Experimental::Detail::RDrawStorage::RPageBoxClicked()
{
   int event = gPad->GetEvent();
   if (event != kButton1Up)
      return;
   TObject *select = gPad->GetSelected();
   if (!select)
      return;
   if (select->InheritsFrom(ROOT::Experimental::Detail::RPageBox::Class())) {
      ROOT::Experimental::Detail::RPageBox *pageBox = (ROOT::Experimental::Detail::RPageBox *)select;
      pageBox->Inspect();
   } else if (select->InheritsFrom(ROOT::Experimental::Detail::RMetaDataBox::Class())) {
      ROOT::Experimental::Detail::RMetaDataBox *metaBox = (ROOT::Experimental::Detail::RMetaDataBox *)select;
      metaBox->Inspect();
   }
}

std::int32_t ROOT::Experimental::Detail::RDrawStorage::GetColourFromFieldId(ROOT::Experimental::DescriptorId_t fieldId)
{
   std::int32_t colour = 0;
   fieldId %= 61;
   switch (fieldId % 12) {
   case 0: colour = kRed; break;
   case 1: colour = kMagenta; break;
   case 2: colour = kBlue; break;
   case 3: colour = kCyan; break;
   case 4: colour = kGreen; break;
   case 5: colour = kYellow; break;
   case 6: colour = kPink; break;
   case 7: colour = kViolet; break;
   case 8: colour = kAzure; break;
   case 9: colour = kTeal; break;
   case 10: colour = kSpring; break;
   case 11: colour = kOrange; break;
   default:
      // never here
      assert(false);
      break;
   }
   switch (fieldId / 12) {
   case 0: colour -= 2; break;
   case 1: break;
   case 2: colour += 3; break;
   case 3: colour -= 6; break;
   case 4: colour -= 9; break;
   case 5:
      if (fieldId == 60)
         return kGray;
   default:
      // never here
      assert(false);
      break;
   }
   return colour;
}

// ------------------------------ RNTupleDraw -------------------------------

ROOT::Experimental::RNTupleDraw::RNTupleDraw(const std::unique_ptr<RNTupleReader> &reader)
{
   if (reader == nullptr) {
      std::cout << "The RNTupleReader is invalid! Drawing is not possible with this object." << std::endl;
      fEmpty = true;
   }
   Detail::RDrawStorage::fgDrawStorageVec.emplace_back(reader.get());
   fStorage = &Detail::RDrawStorage::fgDrawStorageVec.back();
}

std::unique_ptr<ROOT::Experimental::RNTupleDraw>
ROOT::Experimental::RNTupleDraw::Open(const std::unique_ptr<RNTupleReader> &reader)
{
   if (reader == nullptr) {
      std::cout << "The RNTupleReader is invalid, a nullptr was returned." << std::endl;
      return nullptr;
   }
   return std::make_unique<RNTupleDraw>(reader);
}

void ROOT::Experimental::RNTupleDraw::Draw()
{
   if (fEmpty == false) {
      fStorage->Draw();
   } else {
      std::cout << "Cannot draw object from an empty storage." << std::endl;
   }
}
