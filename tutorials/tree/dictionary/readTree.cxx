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

void readTree()
{

   std::unique_ptr<TFile> ifile{TFile::Open("testFile.root", "read")};
   if (!ifile || ifile->IsZombie()) {
      std::cerr << "Problem opening the file testFile.root" << std::endl;
      return;
   }

   // Create a TTreeReader to read the tree named "myTree"
   TTreeReader aReader("myTree", ifile.get());

   // Create TTreeReaderValues for the branches "branch1" and "branch2"
   TTreeReaderValue<myDetectorData> branch1(aReader, "branch1.");
   TTreeReaderValue<myDetectorData> branch2(aReader, "branch2.");

   // Loop over the entries of the tree
   while (aReader.Next()) {
      if (branch1->time != 0)
         std::cout << " -Branch1 : time: " << branch1->time << "\t energy: " << branch1->energy << std::endl;
      else if (branch2->time != 0)
         std::cout << " +Branch2 : time: " << branch2->time << "\t energy: " << branch2->energy << std::endl;
      else
         std::cerr << "WARNING: entry " << aReader.GetCurrentEntry() << " is empty! " << std::endl;
   }
}
