// ##########################################################################
// # Alvaro Tolosa Delgado @ IFIC(Valencia,Spain)  alvaro.tolosa@ific.uv.es #
// # Copyright (c) 2018 Alvaro Tolosa. All rights reserved.		 #
// ##########################################################################

#include <iostream>

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include "data2Tree.hpp"

void readTree()
{

   std::unique_ptr<TFile> ifile = std::make_unique<TFile>("testFile.root", "read");
   if ( nullptr == ifile ) {
      std::cerr << " File not found " << std::endl;
      return;
   }
   if (!TClass::GetClass(typeid(myDetectorData))->IsLoaded()) {
      std::cerr << " TClass::GetClass(typeid(myDetectorData))->IsLoaded() == false " << std::endl;
   }

   // Create a TTreeReader to read the tree named "myTree"
   TTreeReader aReader("myTree", ifile.get() );

   // Create TTreeReaderValues for the branches "branch1" and "branch2"
   TTreeReaderValue<myDetectorData> branch1(aReader, "branch1.");
   TTreeReaderValue<myDetectorData> branch2(aReader, "branch2.");

   // Loop over the entries of the tree
   while (aReader.Next()) {
      if (branch1->time != 0)
         std::cerr << " -Branch1 : time: " << branch1->time << "\t energy: " << branch1->energy << std::endl;
      else if (branch2->time != 0)
         std::cerr << " +Branch2 : time: " << branch2->time << "\t energy: " << branch2->energy << std::endl;
      else
         std::cerr << "WARNING: entry " << aReader.GetCurrentEntry() << " is empty! " << std::endl;
   }

   ifile->Close();
}

