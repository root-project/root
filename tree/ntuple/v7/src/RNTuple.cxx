/// \file RNTuple.cxx
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

#include "ROOT/RNTuple.hxx"

#include "ROOT/RFieldVisitor.hxx"
#include "ROOT/RNTupleModel.hxx"
#include "ROOT/RPageStorage.hxx"
#include "ROOT/RPageStorageRoot.hxx"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

ROOT::Experimental::Detail::RNTuple::RNTuple(std::unique_ptr<ROOT::Experimental::RNTupleModel> model)
   : fModel(std::move(model))
   , fNEntries(0)
{
}

ROOT::Experimental::Detail::RNTuple::~RNTuple()
{
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RNTupleReader::ConnectModel() {
   std::unordered_map<const Detail::RFieldBase *, DescriptorId_t> fieldPtr2Id;
   fieldPtr2Id[fModel->GetRootField()] = fSource->GetDescriptor().FindFieldId("", kInvalidDescriptorId);
   for (auto& field : *fModel->GetRootField()) {
      auto parentId = fieldPtr2Id[field.GetParent()];
      auto fieldId = fSource->GetDescriptor().FindFieldId(field.GetName(), parentId);
      R__ASSERT(fieldId != kInvalidDescriptorId);
      fieldPtr2Id[&field] = fieldId;
      Detail::RFieldFuse::Connect(fieldId, *fSource, field);
   }
}

ROOT::Experimental::RNTupleReader::RNTupleReader(
   std::unique_ptr<ROOT::Experimental::RNTupleModel> model,
   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> source)
   : ROOT::Experimental::Detail::RNTuple(std::move(model))
   , fSource(std::move(source))
{
   fSource->Attach();
   ConnectModel();
   fNEntries = fSource->GetNEntries();
}

ROOT::Experimental::RNTupleReader::RNTupleReader(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> source)
   : ROOT::Experimental::Detail::RNTuple(nullptr)
   , fSource(std::move(source))
{
   fSource->Attach();
   fModel = fSource->GetDescriptor().GenerateModel();
   ConnectModel();
   fNEntries = fSource->GetNEntries();
}

ROOT::Experimental::RNTupleReader::~RNTupleReader()
{
   // needs to be destructed before the page source
   fModel = nullptr;
}

std::unique_ptr<ROOT::Experimental::RNTupleReader> ROOT::Experimental::RNTupleReader::Open(
   std::unique_ptr<RNTupleModel> model,
   std::string_view ntupleName,
   std::string_view storage)
{
   // TODO(jblomer): heuristics based on storage
   return std::make_unique<RNTupleReader>(
      std::move(model), std::make_unique<Detail::RPageSourceRoot>(ntupleName, storage));
}

std::unique_ptr<ROOT::Experimental::RNTupleReader> ROOT::Experimental::RNTupleReader::Open(
   std::string_view ntupleName,
   std::string_view storage)
{
   return std::make_unique<RNTupleReader>(std::make_unique<Detail::RPageSourceRoot>(ntupleName, storage));
}

void ROOT::Experimental::RNTupleReader::PrintInfo(const ENTupleInfo what, std::ostream &output)
{
   // TODO(lesimon): In a later version, these variables may be defined by the user or the ideal width may be read out from the terminal.
   char frameSymbol = '*';
   int width = 80;
   /*
   if (width < 30) {
      output << "The width is too small! Should be at least 30." << std::endl;
      return;
   }
   */
   std::string name = fSource->GetDescriptor().GetName();
   //prepVisitor traverses through all fields to gather information needed for printing.
   RPrepareVisitor prepVisitor;
   //printVisitor traverses through all fields to do the actual printing.
   RPrintVisitor printVisitor(output);
   switch (what) {
   case ENTupleInfo::kSummary:
      for (int i = 0; i < (width/2 + width%2 - 4); ++i)
            output << frameSymbol;
      output << " NTUPLE ";
      for (int i = 0; i < (width/2 - 4); ++i)
         output << frameSymbol;
      output << std::endl;
      // FitString defined in RFieldVisitor.cxx
         output << frameSymbol << " N-Tuple : " << RNTupleFormatter::FitString(name, width-13) << frameSymbol << std::endl; // prints line with name of ntuple
         output << frameSymbol << " Entries : " << RNTupleFormatter::FitString(std::to_string(GetNEntries()), width - 13) << frameSymbol << std::endl;  // prints line with number of entries
      GetModel()->GetRootField()->TraverseVisitor(prepVisitor);

      printVisitor.SetFrameSymbol(frameSymbol);
      printVisitor.SetWidth(width);
      printVisitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
      printVisitor.SetNumFields(prepVisitor.GetNumFields());
      GetModel()->GetRootField()->TraverseVisitor(printVisitor);

      for (int i = 0; i < width; ++i)
         output << frameSymbol;
      output << std::endl;
      return;
   default:
      // Unhandled case, internal error
      assert(false);
   }
}
//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleWriter::RNTupleWriter(
   std::unique_ptr<ROOT::Experimental::RNTupleModel> model,
   std::unique_ptr<ROOT::Experimental::Detail::RPageSink> sink)
   : ROOT::Experimental::Detail::RNTuple(std::move(model))
   , fSink(std::move(sink))
   , fClusterSizeEntries(kDefaultClusterSizeEntries)
   , fLastCommitted(0)
{
   fSink->Create(*fModel.get());
}

ROOT::Experimental::RNTupleWriter::~RNTupleWriter()
{
   CommitCluster();
   fSink->CommitDataset();
   // needs to be destructed before the page sink
   fModel = nullptr;
}


std::unique_ptr<ROOT::Experimental::RNTupleWriter> ROOT::Experimental::RNTupleWriter::Recreate(
   std::unique_ptr<RNTupleModel> model,
   std::string_view ntupleName,
   std::string_view storage)
{
   // TODO(jblomer): heuristics based on storage
   TFile *file = TFile::Open(std::string(storage).c_str(), "RECREATE");
   Detail::RPageSinkRoot::RSettings settings;
   settings.fFile = file;
   settings.fTakeOwnership = true;
   return std::make_unique<RNTupleWriter>(
      std::move(model), std::make_unique<Detail::RPageSinkRoot>(ntupleName, settings));
}


void ROOT::Experimental::RNTupleWriter::CommitCluster()
{
   if (fNEntries == fLastCommitted) return;
   for (auto& field : *fModel->GetRootField()) {
      field.Flush();
      field.CommitCluster();
   }
   fSink->CommitCluster(fNEntries);
   fLastCommitted = fNEntries;
}


//------------------------------------------------------------------------------


ROOT::Experimental::RCollectionNTuple::RCollectionNTuple(std::unique_ptr<REntry> defaultEntry)
   : fOffset(0), fDefaultEntry(std::move(defaultEntry))
{
}
