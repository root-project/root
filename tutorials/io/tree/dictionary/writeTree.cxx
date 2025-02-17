// Author: Alvaro Tolosa-Delgado CERN 07/2023
// Author: Jorge Agramunt Ros IFIC(Valencia,Spain) 07/2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include "data2Tree.hxx"

void writeTree()
{
   std::unique_ptr<TFile> ofile{TFile::Open("testFile.root", "recreate")};
   if (!ofile || ofile->IsZombie()) {
      throw std::runtime_error("Could not open file testFile.root");
   }

   std::unique_ptr<TTree> myTree = std::make_unique<TTree>("myTree", "");
   myDetectorData obj_for_branch1;
   myDetectorData obj_for_branch2;

   // NOTE: the dot at the end of the branch name is semantically relevant and recommended
   // because it causes the sub-branch names to be prefixed by the name of the top level branch.
   // Without the dot, the prefix is not there.
   // Here, objects of the same class appear in multiple branches, adding the dot removes ambiguities.
   myTree->Branch("branch1.", &obj_for_branch1);
   myTree->Branch("branch2.", &obj_for_branch2);

   for (int i = 0; i < 10; ++i) {
      //-- if i is even, fill branch2 and set branch1's entry to zero
      if (i % 2 == 0) {
         obj_for_branch1.clear();
         obj_for_branch2.time = i + 5;
         obj_for_branch2.energy = 2 * i + 5;
         obj_for_branch2.detectorID = 3 * i + 5;
      }
      //-- if i is odd, we do the opposite
      else {
         obj_for_branch2.clear();
         obj_for_branch1.time = i + 1;
         obj_for_branch1.energy = 2 * i + 1;
         obj_for_branch1.detectorID = 3 * i + 1;
      }
      myTree->Fill();
   }

   myTree->Print();

   ofile->Write(); // This write the files and the TTree
}
