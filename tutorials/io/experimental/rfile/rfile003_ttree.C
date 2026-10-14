/// \file
/// \ingroup ROOT7 tutorial_io
/// Demonstrate how to read a TTree with RFile.
///
/// Note that, due to how TTree handles lifetime and its relationship with TDirectory, writing TTree through RFile
/// is currently unsupported.
///
/// \macro_code
/// \macro_output
///
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
///
/// \date 2026-10-12
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
#include <iostream>
#include <TSystem.h>
#include <TTree.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RFile.hxx>

void read_tree_from_rfile(const char *fileName)
{
   // Open an existing ROOT file for reading (will throw an exception if `fileName` cannot be read).
   // IMPORTANT! This file must remain open while reading data from the tree.
   auto file = ROOT::Experimental::RFile::Open(fileName);

   // Get back the TTree from the file
   auto tree = file->Get<TTree>("tree");
   R__ASSERT(tree); // we just wrote the tree so we expect to find it in the file
   int intVal;
   tree->SetBranchAddress("intBranch", &intVal);

   for (auto i = 0; i < tree->GetEntries(); ++i) {
      tree->GetEntry(i);
      std::cout << "value[" << i << "] = " << intVal << "\n";
   }
}

void rfile003_ttree()
{
   const char *const fileName = "rfile_ttree.root";

   // Note that TTree cannot be written to RFile, so we use RDataFrame to write it instead.
   ROOT::RDataFrame(10).Define("intBranch", "static_cast<int>(rdfentry_)").Snapshot("tree", fileName);

   read_tree_from_rfile(fileName);

   gSystem->Unlink(fileName);
}
