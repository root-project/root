// @(#)root/thread:$Id$
// Author: Enric Tejedor, CERN  12/09/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class ROOT::TTreeProcessorMT
    \brief A class to process the entries of a TTree in parallel.

By means of its Process method, ROOT::TTreeProcessorMT provides a way to process the
entries of a TTree in parallel. When invoking TTreeProcessor::Process, the user
passes a function whose only parameter is a TTreeReader. The function iterates
on a subrange of entries by using that TTreeReader.

The implementation of ROOT::TTreeProcessorMT parallelizes the processing of the subranges,
each corresponding to a cluster in the TTree. This is possible thanks to the use
of a ROOT::TThreadedObject, so that each thread works with its own TFile and TTree
objects.
*/

#include "TROOT.h"
#include "ROOT/TTreeProcessorMT.hxx"
#include "ROOT/TThreadExecutor.hxx"

using namespace ROOT;

////////////////////////////////////////////////////////////////////////
/// Constructor based on a file name.
/// \param[in] filename Name of the file containing the tree to process.
/// \param[in] treename Name of the tree to process. If not provided,
///                     the implementation will automatically search for a
///                     tree in the file.
TTreeProcessorMT::TTreeProcessorMT(std::string_view filename, std::string_view treename) : treeView(filename, treename) {}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a collection of file names.
/// \param[in] filenames Collection of the names of the files containing the tree to process.
/// \param[in] treename Name of the tree to process. If not provided,
///                     the implementation will automatically search for a
///                     tree in the collection of files.
TTreeProcessorMT::TTreeProcessorMT(const std::vector<std::string_view> &filenames, std::string_view treename) : treeView(filenames, treename) {}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a TTree.
/// \param[in] tree Tree or chain of files containing the tree to process.
TTreeProcessorMT::TTreeProcessorMT(TTree &tree) : treeView(tree) {}

////////////////////////////////////////////////////////////////////////
/// Constructor based on a TTree and a TEntryList.
/// \param[in] tree Tree or chain of files containing the tree to process.
/// \param[in] entries List of entry numbers to process.
TTreeProcessorMT::TTreeProcessorMT(TTree &tree, TEntryList &entries) : treeView(tree, entries) {}

//////////////////////////////////////////////////////////////////////////////
/// Process the entries of a TTree in parallel. The user-provided function
/// receives a TTreeReader which can be used to iterate on a subrange of
/// entries
/// ~~~{.cpp}
/// TTreeProcessorMT::Process([](TTreeReader& readerSubRange) {
///                            // Select branches to read
///                            while (readerSubRange.next()) {
///                                // Use content of current entry
///                            }
///                         });
/// ~~~
/// The user needs to be aware that each of the subranges can potentially
/// be processed in parallel. This means that the code of the user function
/// should be thread safe.
///
/// \param[in] func User-defined function that processes a subrange of entries
void TTreeProcessorMT::Process(std::function<void(TTreeReader &)> func)
{
   // Enable this IMT use case (activate its locks)
   Internal::TParTreeProcessingRAII ptpRAII;

   // Iterate over the collection of files
   std::vector<std::tuple<Long64_t, Long64_t, size_t>> vTuple;
   for (size_t i = 0; i < treeView->GetNumFiles(); ++i) {
      treeView->SetCurrent(i);
      auto clusterIter = treeView->GetClusterIterator();
      Long64_t start = 0, end = 0;
      // Iterate over the clusters in the current file and generate a task for each of them
      while ((start = clusterIter()) < treeView->GetEntries()) {
         end = clusterIter.GetNextEntry();
         vTuple.emplace_back(start, end, i);
      }
   }

   auto mapFunction = [this, &func](const std::tuple<Long64_t, Long64_t, size_t> &t) {
      treeView->SetCurrent(std::get<2>(t));
      auto tr = treeView->GetTreeReader(std::get<0>(t), std::get<1>(t));
      func(*tr);
   };

   // Assume number of threads has been initialized via ROOT::EnableImplicitMT
   TThreadExecutor pool;
   pool.Foreach(mapFunction, vTuple);
}
