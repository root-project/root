// @(#)root/treeplayer:$Id$
// Author: Axel Naumann, 2011-09-21

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTreeReader.h"

#include "TChain.h"
#include "TDirectory.h"
#include "TTreeReaderValue.h"

/** \class TTreeReader
 TTreeReader is a simple, robust and fast interface to read values from a TTree,
 TChain or TNtuple.

 It uses TTreeReaderValue<T> and TTreeReaderArray<T> to access the data.

 Example code can be found in
 tutorials/tree/hsimpleReader.C and tutorials/trees/h1analysisTreeReader.h and
 tutorials/trees/h1analysisTreeReader.C for a TSelector.

 Roottest contains an
 <a href="http://root.cern.ch/gitweb?p=roottest.git;a=tree;f=root/tree/reader;hb=HEAD">example</a>
 showing the full power.

A simpler analysis example - the one from the tutorials - can be found below:
it histograms a function of the px and py branches.

~~~{.cpp}
// A simple TTreeReader use: read data from hsimple.root (written by hsimple.C)

#include "TFile.h
#include "TH1F.h
#include "TTreeReader.h
#include "TTreeReaderValue.h

void hsimpleReader() {
   // Create a histogram for the values we read.
   TH1F("h1", "ntuple", 100, -4, 4);

   // Open the file containing the tree.
   TFile *myFile = TFile::Open("$ROOTSYS/tutorials/hsimple.root");

   // Create a TTreeReader for the tree, for instance by passing the
   // TTree's name and the TDirectory / TFile it is in.
   TTreeReader myReader("ntuple", myFile);

   // The branch "px" contains floats; access them as myPx.
   TTreeReaderValue<Float_t> myPx(myReader, "px");
   // The branch "py" contains floats, too; access those as myPy.
   TTreeReaderValue<Float_t> myPy(myReader, "py");

   // Loop over all entries of the TTree or TChain.
   while (myReader.Next()) {
      // Just access the data as if myPx and myPy were iterators (note the '*'
      // in front of them):
      myHist->Fill(*myPx + *myPy);
   }

   myHist->Draw();
}
~~~

A more complete example including error handling and a few combinations of
TTreeReaderValue and TTreeReaderArray would look like this:

~~~{.cpp}
#include <TFile.h>
#include <TH1.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

#include "TriggerInfo.h"
#include "Muon.h"
#include "Tau.h"

#include <vector>
#include <iostream>

bool CheckValue(ROOT::TTreeReaderValueBase* value) {
   if (value->GetSetupStatus() < 0) {
      std::cerr << "Error " << value->GetSetupStatus()
                << "setting up reader for " << value->GetBranchName() << '\n';
      return false;
   }
   return true;
}


// Analyze the tree "MyTree" in the file passed into the function.
// Returns false in case of errors.
bool analyze(TFile* file) {
   // Create a TTreeReader named "MyTree" from the given TDirectory.
   // The TTreeReader gives access to the TTree to the TTreeReaderValue and
   // TTreeReaderArray objects. It knows the current entry number and knows
   // how to iterate through the TTree.
   TTreeReader reader("MyTree", file);

   // Read a single float value in each tree entries:
   TTreeReaderValue<float> weight(reader, "event.weight");
   if (!CheckValue(weight)) return false;

   // Read a TriggerInfo object from the tree entries:
   TTreeReaderValue<TriggerInfo> triggerInfo(reader, "triggerInfo");
   if (!CheckValue(triggerInfo)) return false;

   //Read a vector of Muon objects from the tree entries:
   TTreeReaderValue<std::vector<Muon>> muons(reader, "muons");
   if (!CheckValue(muons)) return false;

   //Read the pT for all jets in the tree entry:
   TTreeReaderArray<double> jetPt(reader, "jets.pT");
   if (!CheckValue(jetPt)) return false;

   // Read the taus in the tree entry:
   TTreeReaderArray<Tau> taus(reader, "taus");
   if (!CheckValue(taus)) return false;


   // Now iterate through the TTree entries and fill a histogram.

   TH1F("hist", "TTreeReader example histogram", 10, 0., 100.);

   while (reader.Next()) {

      if (reader.GetEntryStatus() == kEntryValid) {
         std::cout << "Loaded entry " << reader.GetCurrentEntry() << '\n';
      } else {
         switch (reader.GetEntryStatus()) {
         kEntryValid:
            // Handled above.
            break;
         kEntryNotLoaded:
            std::cerr << "Error: TTreeReader has not loaded any data yet!\n";
            break;
         kEntryNoTree:
            std::cerr << "Error: TTreeReader cannot find a tree names \"MyTree\"!\n";
            break;
         kEntryNotFound:
            // Can't really happen as TTreeReader::Next() knows when to stop.
            std::cerr << "Error: The entry number doe not exist\n";
            break;
         kEntryChainSetupError:
            std::cerr << "Error: TTreeReader cannot access a chain element, e.g. file without the tree\n";
            break;
         kEntryChainFileError:
            std::cerr << "Error: TTreeReader cannot open a chain element, e.g. missing file\n";
            break;
         kEntryDictionaryError:
            std::cerr << "Error: TTreeReader cannot find the dictionary for some data\n";
            break;
         }
         return false;
      }

      // Access the TriggerInfo object as if it's a pointer.
      if (!triggerInfo->hasMuonL1())
         continue;

      // Ditto for the vector<Muon>.
      if (!muons->size())
         continue;

      // Access the jetPt as an array, whether the TTree stores this as
      // a std::vector, std::list, TClonesArray or Jet* C-style array, with
      // fixed or variable array size.
      if (jetPt.GetSize() < 2 || jetPt[0] < 100)
         continue;

      // Access the array of taus.
      if (!taus.IsEmpty()) {
         float currentWeight = *weight;
         for (int iTau = 0, nTau = taus.GetSize(); iTau < nTau; ++iTau) {
            // Access a float value - need to dereference as TTreeReaderValue
            // behaves like an iterator
            hist->Fill(taus[iTau].eta(), currentWeight);
         }
      }
   } // TTree entry / event loop
}
~~~
*/

ClassImp(TTreeReader)

using namespace ROOT::Internal;

////////////////////////////////////////////////////////////////////////////////
/// Access data from tree.

TTreeReader::TTreeReader(TTree* tree):
   fTree(tree),
   fDirectory(0),
   fEntryStatus(kEntryNotLoaded),
   fDirector(0),
   fLastEntry(-1),
   fProxiesSet(kFALSE)
{
   Initialize();
}

////////////////////////////////////////////////////////////////////////////////
/// Access data from the tree called keyname in the directory (e.g. TFile)
/// dir, or the current directory if dir is NULL. If keyname cannot be
/// found, or if it is not a TTree, IsZombie() will return true.

TTreeReader::TTreeReader(const char* keyname, TDirectory* dir /*= NULL*/):
   fTree(0),
   fDirectory(dir),
   fEntryStatus(kEntryNotLoaded),
   fDirector(0),
   fLastEntry(-1),
   fProxiesSet(kFALSE)
{
   if (!fDirectory) fDirectory = gDirectory;
   fDirectory->GetObject(keyname, fTree);
   Initialize();
}

////////////////////////////////////////////////////////////////////////////////
/// Tell all value readers that the tree reader does not exist anymore.

TTreeReader::~TTreeReader()
{
   for (std::deque<ROOT::Internal::TTreeReaderValueBase*>::const_iterator
           i = fValues.begin(), e = fValues.end(); i != e; ++i) {
      (*i)->MarkTreeReaderUnavailable();
   }
   delete fDirector;
   fProxies.SetOwner();
}

////////////////////////////////////////////////////////////////////////////////
/// Initialization of the director.

void TTreeReader::Initialize()
{
   if (!fTree) {
      MakeZombie();
      fEntryStatus = kEntryNoTree;
   } else {
      fDirector = new ROOT::Internal::TBranchProxyDirector(fTree, -1);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set the range of entries to be processed.
/// If last > first, this call is equivalent to
/// `SetEntry(first); SetLastEntry(last);`. Otherwise `last` is ignored and
/// only `first` is set.
/// \return the EEntryStatus that would be returned by SetEntry(first)

TTreeReader::EEntryStatus TTreeReader::SetEntriesRange(Long64_t first, Long64_t last)
{
   if(last > first)
      fLastEntry = last;
   else
      fLastEntry = -1;
   return SetLocalEntry(first);
}

////////////////////////////////////////////////////////////////////////////////
///Returns the index of the current entry being read

Long64_t TTreeReader::GetCurrentEntry() const {
   if (!fDirector) return 0;
   Long64_t currentTreeEntry = fDirector->GetReadEntry();
   if (fTree->IsA() == TChain::Class() && currentTreeEntry >= 0) {
      return ((TChain*)fTree)->GetChainEntryNumber(currentTreeEntry);
   }
   return currentTreeEntry;
}

////////////////////////////////////////////////////////////////////////////////
/// Load an entry into the tree, return the status of the read.
/// For chains, entry is the global (i.e. not tree-local) entry number.

TTreeReader::EEntryStatus TTreeReader::SetEntryBase(Long64_t entry, Bool_t local)
{
   if (!fTree) {
      fEntryStatus = kEntryNoTree;
      return fEntryStatus;
   }

   TTree* prevTree = fDirector->GetTree();

   Long64_t loadResult;
   if (!local){
      Int_t treeNumInChain = fTree->GetTreeNumber();

      loadResult = fTree->LoadTree(entry);

      if (loadResult == -2) {
         fEntryStatus = kEntryNotFound;
         return fEntryStatus;
      }

      Int_t currentTreeNumInChain = fTree->GetTreeNumber();
      if (treeNumInChain != currentTreeNumInChain) {
            fDirector->SetTree(fTree->GetTree());
      }
   }
   else {
      loadResult = entry;
   }
   if (!prevTree || fDirector->GetReadEntry() == -1 || !fProxiesSet) {
      // Tell readers we now have a tree
      for (std::deque<ROOT::Internal::TTreeReaderValueBase*>::const_iterator
              i = fValues.begin(); i != fValues.end(); ++i) { // Iterator end changes when parameterized arrays are read
         (*i)->CreateProxy();

         if (!(*i)->GetProxy()){
            fEntryStatus = kEntryDictionaryError;
            return fEntryStatus;
         }
      }
      // If at least one proxy was there and no error occurred, we assume the proxies to be set.
      fProxiesSet = !fValues.empty();
   }
   if (fLastEntry >= 0 && loadResult >= fLastEntry) {
      fEntryStatus = kEntryLast;
      return fEntryStatus;
   }
   fDirector->SetReadEntry(loadResult);
   fEntryStatus = kEntryValid;
   return fEntryStatus;
}

////////////////////////////////////////////////////////////////////////////////
/// Set (or update) the which tree to reader from. tree can be
/// a TTree or a TChain.

void TTreeReader::SetTree(TTree* tree)
{
   fTree = tree;
   if (fTree) {
      ResetBit(kZombie);
      if (fTree->InheritsFrom(TChain::Class())) {
         SetBit(kBitIsChain);
      }
   }

   if (!fDirector) {
      Initialize();
   }
   else {
      fDirector->SetTree(fTree);
      fDirector->SetReadEntry(-1);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add a value reader for this tree.

void TTreeReader::RegisterValueReader(ROOT::Internal::TTreeReaderValueBase* reader)
{
   fValues.push_back(reader);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a value reader for this tree.

void TTreeReader::DeregisterValueReader(ROOT::Internal::TTreeReaderValueBase* reader)
{
   std::deque<ROOT::Internal::TTreeReaderValueBase*>::iterator iReader
      = std::find(fValues.begin(), fValues.end(), reader);
   if (iReader == fValues.end()) {
      Error("DeregisterValueReader", "Cannot find reader of type %s for branch %s", reader->GetDerivedTypeName(), reader->fBranchName.Data());
      return;
   }
   fValues.erase(iReader);
}
