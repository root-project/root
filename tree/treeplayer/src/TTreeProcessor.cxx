// @(#)root/thread:$Id$
// Author: Enric Tejedor, CERN  12/09/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TTreeProcessor
    \brief A class to process the entries of a TTree in parallel.
 
By means of its Process method, TTProcessor provides a way to process the
entries of a TTree in parallel. When invoking TTProcessor::Process, the user
passes a function whose only parameter is a TTreeReader. The function iterates
on a subrange of entries by using that TTreeReader.

The implementation of TTreeProcessor parallelizes the processing of the subranges,
each corresponding to a cluster in the TTree. This is possible thanks to the use
of a ROOT::TThreadedObject, so that each thread works with its own TFile and TTree
objects.
*/

#include "TROOT.h"
#include "ROOT/TTreeProcessor.hxx"

#include "tbb/task.h"
#include "tbb/task_group.h"

using namespace ROOT;

//////////////////////////////////////////////////////////////////////////////
/// Process the entries of a TTree in parallel. The user-provided function
/// receives a TTreeReader which can be used to iterate on a subrange of
/// entries
/// ~~~{.cpp}
/// TTreeProcessor::Process([](TTreeReader& readerSubRange) {
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
void TTreeProcessor::Process(std::function<void(TTreeReader&)> func)
{
   // Enable this IMT use case (activate its locks)
   Internal::TParTreeProcessingRAII ptpRAII;

   auto clusterIter = treeView->GetClusterIterator();
   Long64_t start = 0, end = 0;

   // Create task group - assume number of threads has been initialized via ROOT::EnableImplicitMT
   tbb::task_group g;

   // Iterate over the clusters and generate a task for each of them
   while ((start = clusterIter()) < treeView->GetEntries()) {
      end = clusterIter.GetNextEntry();

      g.run([this, &func, start, end]() {
         auto tr = treeView->GetTreeReader();
         tr->SetEntriesRange(start - 1, end);
         func(*tr);
      });
   }

   g.wait();
}
