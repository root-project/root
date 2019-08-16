/// \file ROOT/RNTupleBrowser.cxx
/// \ingroup NTupleBrowse ROOT7
/// \author Simon Leisibach <simon.satoshi.rene.leisibach@cern.ch>
/// \date 2019-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <cstring>
#include <string>

#include <Rtypes.h>

#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTupleBrowser.hxx>
#include <ROOT/RPageStorageRoot.hxx>


//--------------------------- RNTupleBrowser -----------------------------

ClassImp(ROOT::Experimental::RNTupleBrowser);


ROOT::Experimental::RNTupleBrowser::RNTupleBrowser(TDirectory* directory, int unittest): fDirectory{directory}, fReaderPtr{nullptr}, fUnitTest{unittest}, fCurrentTH1F{nullptr}
{
}


void ROOT::Experimental::RNTupleBrowser::SetDirectory(TDirectory* directory)
{
   fDirectory = directory;
   // Checks if a RNTupleReader with that directory was already created. If so it just takes the already existing RNTupleReader.
   for(int i = 0; i < (int)fDirectoryVec.size(); ++i) {
      if (fDirectoryVec.at(i) == fDirectory) {
         fReaderPtr = fReaderPtrVec.at(i);
         return;
      }
   }
   
   std::string fullPath = fDirectory->GetPath();
   // The 5 lines below are used for unit tests in ntuplebrowse.cxx since a TDirectory can't be initialized with a path.
   if ((fUnitTest/1000) % 10 == 1) {
      fullPath = "test.root";
   } else if ((fUnitTest/1000) % 10 == 2) {
      fullPath = "test2.root";
   }
   std::string rootFileName = std::string(fullPath, 0, fullPath.find(".root") + 5);
   
   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> sourcePtr = std::make_unique<ROOT::Experimental::Detail::RPageSourceRoot>(fDirectory->GetName(), rootFileName, fDirectory);
   
   // These 3 lines allow lines 2 to 7 in this function to be executed in the future.
   fReaderPtrVec.emplace_back(std::make_shared<ROOT::Experimental::RNTupleReader>(std::move(sourcePtr)));
   fReaderPtr = fReaderPtrVec.back();
   fDirectoryVec.push_back(fDirectory);
}
   
ROOT::Experimental::RNTupleBrowser::~RNTupleBrowser()
{
   for (auto ptr : fNTupleBrowsePtrVec) {
      delete ptr;
   }
   // No need to delete TH1F* fCurrentTH1F, because it's automatically deallocated when TBrowser is closed.
}

void ROOT::Experimental::RNTupleBrowser::Browse(TBrowser *b)
{
   ROOT::Experimental::RBrowseVisitor browseVisitor(b, this);
   fReaderPtr->GetModel()->GetRootField()->TraverseVisitor(browseVisitor);
}
   
//---------------------------- NTupleFieldElementFolder ---------------------------
   
ClassImp(ROOT::Experimental::RNTupleFieldElementFolder);

void ROOT::Experimental::RNTupleFieldElementFolder::Browse(TBrowser *b)
{
   RBrowseVisitor browseVisitor(b, fRNTupleBrowserPtr);
   if (fFieldPtr) {
      fFieldPtr->TraverseVisitor(browseVisitor);
   } else {
      fNtupleReaderPtr->GetModel()->GetRootField()->TraverseVisitor(browseVisitor);
   }
}

void ROOT::Experimental::RNTupleFieldElementFolder::AddBrowse(TBrowser *b)
{
   // in case of unit test, b->Add(this) will call the Add function implemented in TBrowser.h, despite the pointer not pointing to an instance from TBrowser.h (instead it points to the one created in ntuple_browse.cxx) For that reason, b->Add(this) is not called, when (fUnitTest != 0)<=> program is executed in an unittest.
   if (fRNTupleBrowserPtr->GetfUnitTest()) {
      fRNTupleBrowserPtr->IncreasefUnitTest();
   } else {
      b->Add(this);
   }
}
   
//---------------------------- NTupleFieldElement --------------------
   
ClassImp(ROOT::Experimental::RNTupleFieldElement);

void ROOT::Experimental::RNTupleFieldElement::AddBrowse(TBrowser *b)
{
   if (fRNTupleBrowserPtr->GetfUnitTest()) {
      fRNTupleBrowserPtr->IncreasefUnitTest();
   } else {
      b->Add(this);
   }
}

void ROOT::Experimental::RNTupleFieldElement::Browse(TBrowser* /*b*/)
{
   switch(fType) {
      case fieldDatatype_nonNumeric:
      case fieldDatatype_notkLeaf:
      case fieldDatatype_parentIsVec:
      case fieldDatatype_noHist: break;
      case fieldDatatype_float: TemplatedBrowse<float>(false); break;
      case fieldDatatype_double: TemplatedBrowse<double>(false); break;
      case fieldDatatype_Int32: TemplatedBrowse<std::int32_t>(true); break;
      case fieldDatatype_UInt32: TemplatedBrowse<std::uint32_t>(true); break;
      case fieldDatatype_UInt64: TemplatedBrowse<std::uint64_t>(true); break;
   }
}

template <typename T>
void ROOT::Experimental::RNTupleFieldElement::TemplatedBrowse(bool isIntegraltype)
{
   const char* name = (const char*)fName;
   auto ntupleView = fReaderPtr->GetView<T>(std::string(name));
   auto numEntries{ fReaderPtr->GetNEntries() };
   if (numEntries == 0) return; // no histogram
   
   double max{static_cast<double>(ntupleView(0))}, min{static_cast<double>(ntupleView(0))};
   for(unsigned long long i = 1; i < numEntries; ++i) {
      if (ntupleView(i) > max) {
         max = ntupleView(i);
      } else if (ntupleView(i) < min) {
         min = ntupleView(i);
      }
   }
   // number of bins in histogram shouldn't be 100 if the histogram is only filled with integers in a range of 1-10. To account for such cases, the number of bins is set here.
   int nbins = isIntegraltype ? std::min(max - min+1, 100.0) : 100;
   
   
   TH1F* h1 = new TH1F(name, name, nbins, min - 0.5, max + 0.5);
   for (unsigned long long i = 0; i < numEntries; ++i)
      h1->Fill(ntupleView(i));
   h1->Draw();
   
   
   if (fRNTupleBrowserPtr->fCurrentTH1F) {
      delete fRNTupleBrowserPtr->fCurrentTH1F;
   }
   fRNTupleBrowserPtr->fCurrentTH1F = h1;
}
