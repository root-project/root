// This test is a derivative of what's used in http://cern.ch/root-intro.
// It tests a common user error: re-accessing a TTreeReaderValue after the loop.

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"

bool checkClass(TFile* file) {
   // Create the tree reader and its data containers
   TTreeReader badReader("EventTree", file);
   // None of these match the branch type.
   TTreeReaderArray<Int_t> thisIsWrong1(badReader, "fParticles");
   TTreeReaderValue<Int_t> thisIsWrong2(badReader, "fParticles");

   while (badReader.Next()) {}

   // All setup errors are < 0.
   int setupStatus = (int)thisIsWrong2.GetSetupStatus();
   return (setupStatus < 0);
}

bool checkLeaf1(TFile* file) {
   // Create the tree reader and its data containers
   TTreeReader badReader("EventTree", file);
   // Must use TTreeReaderValueArray to access a member of an object that is stored in a collection.
   TTreeReaderValue<double> thisIsWrong3(badReader, "fParticles.fMomentum");

   while (badReader.Next()) {}

   // All setup errors are < 0.
   int setupStatus = (int)thisIsWrong3.GetSetupStatus();
   return (setupStatus < 0);
}


// The readers do not all report mismatches in one go - which is fine because
// they will do in the end, if any mismatch is remaining. Thus split the tests.
bool checkLeaf2(TFile* file) {
   // Create the tree reader and its data containers
   TTreeReader badReader("EventTree", file);

   // The branch fParticles.fPosX contains data of type double. It cannot be accessed by a TTreeReaderArray<float>
   TTreeReaderArray<float> thisIsWrong4(badReader, "fParticles.fPosX");

   // The tree does not have a branch called fDoesntExist. You could check with TTree::Print() for available branches.
   TTreeReaderArray<int> thisIsWrong5(badReader, "fDoesntExist");

   while (badReader.Next()) {}

   // All setup errors are < 0.
   int setupStatus = (int)thisIsWrong4.GetSetupStatus();
   return (setupStatus < 0);
}

int execIntroTut()
{
   // open the file
   const auto fname = "root://eospublic.cern.ch//eos/root-eos/testfiles/eventdata.root";
   TFile *file = TFile::Open(fname);
   if (file == 0) {
      // if we cannot open the file, print an error message and return immediatly
      printf("Error: cannot open %s!\n", fname);
      return 42;
   }

   int ret = 0;
   if (!checkClass(file))
      ret += 1;
   if (!checkLeaf1(file))
      ret += 2;
   if (!checkLeaf2(file))
      ret += 4;

   return ret;
}
