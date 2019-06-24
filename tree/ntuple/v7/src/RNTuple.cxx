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
#include <iostream>

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
   fieldPtr2Id[fModel->GetRootField()] = kInvalidDescriptorId;
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

std::string ROOT::Experimental::RNTupleReader::GetInfo(const ENTupleInfo what) {
   std::ostringstream os;
   auto name = fSource->GetDescriptor().GetName();

   switch (what) {
   case ENTupleInfo::kSummary:
      os << "****************************** NTUPLE ******************************"  << std::endl
         << "* Name:    " << name << std::setw(57 - name.length())           << "*" << std::endl
         << "* Entries: " << std::setw(10) << fNEntries << std::setw(47)     << "*" << std::endl
         << "********************************************************************"  << std::endl;
      return os.str();
   default:
      // Unhandled case, internal error
      assert(false);
   }
   // Never here
   return "";
}


void ROOT::Experimental::RNTupleReader::Print(std::ostream &output, char frameSymbol, int width)
{
   if (width < 30) {
      std::cout << "The width is too small! Should be at least 30.\n";
      return;
   }
   for (int i = 0; i < (width/2 + width%2 - 4); ++i) output << frameSymbol; output << " NTUPLE ";
   for (int i = 0; i < (width/2 - 4); ++i) output << frameSymbol; output << '\n';
   //CutStringAndAddEllipsisIfNeeded defined in RFieldVisitor.hxx
   output << frameSymbol << " Ntuple  : " << CutStringAndAddEllipsisIfNeeded(GetName(), width-13) << std::string(std::max(0, static_cast<int>(width-13-GetName().size())), ' ' ) << frameSymbol << '\n'; // prints line with name of ntuple
   output << frameSymbol << " Entries : " << GetNEntries() << std::string(std::max(0, static_cast<int>(width-13-(std::to_string(GetNEntries())).size())), ' ') << frameSymbol << '\n';  // prints line with number of entries
   
   //prepVisitor traverses through all fields to gather information needed for printing.
   RPrepareVisitor prepVisitor;
   GetModel()->GetRootField()->TraverseVisitor(prepVisitor);
   
   //printVisitor traverses through all fields to do the actual printing.
   RPrintVisitor printVisitor(output);
   //To make code more understandable, all the parameter setting is done here instead of inside the constructor
   printVisitor.SetFrameSymbol(frameSymbol);
   printVisitor.SetWidth(width);
   printVisitor.SetDeepestLevel(prepVisitor.getDeepestLevel());
   printVisitor.SetNumFields(prepVisitor.getNumFields());
   printVisitor.SetAvailableSpaceForStrings();
   printVisitor.ResizeFlagVec();
   GetModel()->GetRootField()->TraverseVisitor(printVisitor);
   
   
   for(int i = 0; i < width; ++i) output << frameSymbol; output << '\n';
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
