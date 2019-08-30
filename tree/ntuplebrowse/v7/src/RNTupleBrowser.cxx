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

#include <ROOT/RBrowseVisitor.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTupleBrowser.hxx>
#include <ROOT/RPageStorageRoot.hxx>


#include <Rtypes.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>

//--------------------------- RNTupleBrowser -----------------------------

ROOT::Experimental::RNTupleBrowser::RNTupleBrowser(TDirectory* directory): fDirectory{directory}, fCurrentTH1F{nullptr}
{
}


void ROOT::Experimental::RNTupleBrowser::SetDirectory(TDirectory* directory)
{
   fDirectory = directory;
   // Checks if a RNTupleReader with that directory was already created. If so it just takes the already existing RNTupleReader.
   for (unsigned int i = 0; i < fPastDirectories.size(); ++i) {
      if (fPastDirectories.at(i) == fDirectory) {
         fReaderPtrVecIndex = i;
         return;
      }
   }
   
   std::unique_ptr<Detail::RPageSource> sourcePtr = std::make_unique<Detail::RPageSourceRoot>(fDirectory);
   
   // Stores smart pointers of RNTupleReader in a vector, so that RNTupleReader-objects don't get destroyed at the end of this function. Because the fDirectory serves like a key to the RNTupleReader (like in a map data structure), it is also stored in a vector.
   fReaderPtrVec.emplace_back(std::make_unique<ROOT::Experimental::RNTupleReader>(std::move(sourcePtr)));
   fReaderPtrVecIndex = static_cast<int>(fReaderPtrVec.size()-1);
   fPastDirectories.push_back(fDirectory);
}
   
ROOT::Experimental::RNTupleBrowser::~RNTupleBrowser()
{
   // No need to delete TH1F* fCurrentTH1F, because it's automatically deallocated when TBrowser is closed.
}

void ROOT::Experimental::RNTupleBrowser::Browse(TBrowser *b)
{
   ROOT::Experimental::RBrowseVisitor browseVisitor(b, this);
   GetReaderPtr()->GetModel()->GetRootField()->TraverseVisitor(browseVisitor, 0);
}
   
//---------------------------- NTupleBrowseFolder ---------------------------
   
ClassImp(ROOT::Experimental::RNTupleBrowseFolder);

void ROOT::Experimental::RNTupleBrowseFolder::Browse(TBrowser *b)
{
   RBrowseVisitor browseVisitor(b, fRNTupleBrowserPtr);
   if (fFieldPtr) {
      fFieldPtr->TraverseVisitor(browseVisitor, 0);
   }
}

void ROOT::Experimental::RNTupleBrowseFolder::AddBrowse(TBrowser *b)
{
   // The if statement allows nothing to happen in this function when this function is called in a unit test.
   if (b != nullptr ) {
      b->Add(this);
   }
}
   
//---------------------------- RNTupleBrowseLeaf --------------------

ClassImp(ROOT::Experimental::RNTupleBrowseLeaf);

void ROOT::Experimental::RNTupleBrowseLeaf::AddBrowse(TBrowser *b)
{
   // The if statement allows nothing to happen in this function when this function is called in a unit test.
   if (b != nullptr) {
      b->Add(this);
   }
}

void ROOT::Experimental::RNTupleBrowseLeaf::Browse(TBrowser* /*b*/)
{
   RDisplayHistVisitor histVisitor(fRNTupleBrowserPtr, fReaderPtr);
   if (fFieldPtr) {
      fFieldPtr->AcceptVisitor(histVisitor, 1/* 1 is a dummy value*/);
   }
}
