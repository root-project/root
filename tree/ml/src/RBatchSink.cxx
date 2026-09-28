// Author: Silia Taider, CERN 08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/ML/RBatchSink.hxx"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "ROOT/ML/RFlat2DMatrix.hxx"
#include "ROOT/RNTupleModel.hxx"
#include "ROOT/RNTupleWriter.hxx"
#include "TFile.h"
#include "TTree.h"

namespace ROOT::Experimental::Internal::ML {

void RBatchSink::FillBatch(const RFlat2DMatrix &batch)
{
   if (batch.GetCols() != fRowWidth)
      throw std::runtime_error("RBatchSink::FillBatch: batch has " + std::to_string(batch.GetCols()) +
                               " columns, expected " + std::to_string(fRowWidth));

   const float *data = batch.GetData();
   for (std::size_t row = 0; row < batch.GetRows(); row++)
      FillRow(data + row * fRowWidth);
}

RTTreeBatchSink::RTTreeBatchSink(std::string_view dataset_name, std::string_view filename,
                                 std::vector<RColumnLayout> layout)
   : RBatchSink(std::move(layout)), fFile(TFile::Open(std::string(filename).c_str(), "RECREATE"))
{
   fTree = new TTree(std::string(dataset_name).c_str(), std::string(dataset_name).c_str());
   fTree->SetDirectory(fFile.get()); // fFile owns the TTree

   fRow.resize(fRowWidth);
   fVectorBuffers.resize(
      std::count_if(fLayout.begin(), fLayout.end(), [](const RColumnLayout &col) { return col.fIsVector; }));

   std::size_t vecIdx = 0;
   for (const auto &col : fLayout) {
      if (col.fIsVector) {
         fVectorBuffers[vecIdx].resize(col.fWidth);
         fTree->Branch(col.fName.c_str(), &fVectorBuffers[vecIdx]);
         vecIdx++;
      } else {
         fTree->Branch(col.fName.c_str(), fRow.data() + col.fOffset, (col.fName + "/F").c_str());
      }
   }
}

RTTreeBatchSink::~RTTreeBatchSink() = default;

void RTTreeBatchSink::FillRow(const float *row)
{
   std::copy_n(row, fRowWidth, fRow.begin());

   std::size_t vecIdx = 0;
   for (const auto &col : fLayout) {
      if (col.fIsVector)
         std::copy_n(row + col.fOffset, col.fWidth, fVectorBuffers[vecIdx++].begin());
   }

   fTree->Fill();
}

void RTTreeBatchSink::Commit()
{
   fFile->Write();
}

RNTupleBatchSink::RNTupleBatchSink(std::string_view dataset_name, std::string_view filename,
                                   std::vector<RColumnLayout> layout)
   : RBatchSink(std::move(layout))
{
   auto model = ROOT::RNTupleModel::Create();
   fDestinations.reserve(fLayout.size());

   for (const auto &col : fLayout) {
      if (col.fIsVector) {
         auto field = model->MakeField<std::vector<float>>(col.fName);
         field->resize(col.fWidth);
         fDestinations.push_back(field->data());
         fVectorFields.push_back(std::move(field));
      } else {
         auto field = model->MakeField<float>(col.fName);
         fDestinations.push_back(field.get());
         fScalarFields.push_back(std::move(field));
      }
   }

   fWriter = ROOT::RNTupleWriter::Recreate(std::move(model), std::string(dataset_name), std::string(filename));
}

RNTupleBatchSink::~RNTupleBatchSink() = default;

void RNTupleBatchSink::FillRow(const float *row)
{
   for (std::size_t i = 0; i < fLayout.size(); i++)
      std::copy_n(row + fLayout[i].fOffset, fLayout[i].fWidth, fDestinations[i]);
   fWriter->Fill();
}

void RNTupleBatchSink::Commit()
{
   fWriter->CommitDataset();
}

std::unique_ptr<RBatchSink> CreateBatchSink(std::string_view dataset_name, std::string_view filename,
                                            std::vector<RColumnLayout> layout, std::string_view format)
{
   if (format == "ttree")
      return std::make_unique<RTTreeBatchSink>(dataset_name, filename, std::move(layout));
   if (format == "rntuple")
      return std::make_unique<RNTupleBatchSink>(dataset_name, filename, std::move(layout));

   throw std::runtime_error("CreateBatchSink: unrecognised output format \"" + std::string(format) +
                            "\", expected \"ttree\" or \"rntuple\"");
}

} // namespace ROOT::Experimental::Internal::ML
