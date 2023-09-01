/// \file
/// \ingroup tutorial_net
/// Client program which creates and fills 2 histograms and a TTree.
/// Every 1000000 fills the histograms and TTree is send to the server which displays the histogram.
///
/// To run this demo do the following:
///   - Open at least 2 windows
///   - Start ROOT in the first windows
///   - Execute in the first window: .x fastMergeServer.C
///   - Execute in the other windows: root.exe -b -l -q .x treeClient.C
///     (You can put it in the background if wanted).
/// If you want to run the hserv.C on a different host, just change
/// "localhost" in the TSocket ctor below to the desired hostname.
///
/// \macro_code
///
/// \authors Fons Rademakers, Philippe Canal

#include "TMessage.h"
#include "TBenchmark.h"
#include "TSocket.h"
#include "TH2.h"
#include "TTree.h"
#include "TParallelMergingFile.h"
#include "TRandom.h"
#include "TError.h"

void parallelMergeClient()
{
   gBenchmark->Start("treeClient");

   TParallelMergingFile *file = (TParallelMergingFile*)TFile::Open("mergedClient.root?pmerge=localhost:1095","RECREATE");

   file->Write();
   file->UploadAndReset();       // We do this early to get assigned an index.
   UInt_t idx = file->fServerIdx; // This works on in ACLiC.

   TH1 *hpx;
   if (idx%2 == 0) {
      // Create the histogram
      hpx = new TH1F("hpx","This is the px distribution",100,-4,4);
      hpx->SetFillColor(48);  // set nice fill-color
   } else {
      hpx = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
   }
   Float_t px, py;
   TTree *tree = new TTree("tree","tree");
   tree->SetAutoFlush(4000000);
   tree->Branch("px",&px);
   tree->Branch("py",&py);

   // Fill histogram randomly
   gRandom->SetSeed();
   const int kUPDATE = 1000000;
   for (int i = 0; i < 25000000; ) {
      gRandom->Rannor(px,py);
      if (idx%2 == 0)
         hpx->Fill(px);
      else
         hpx->Fill(px,py);
      tree->Fill();
      ++i;
      if (i && (i%kUPDATE) == 0) {
         file->Write();
      }
   }
   file->Write();
   delete file;

   gBenchmark->Show("treeClient");
}
