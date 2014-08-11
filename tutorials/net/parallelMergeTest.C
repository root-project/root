#include "TMessage.h"
#include "TBenchmark.h"
#include "TSocket.h"
#include "TH2.h"
#include "TTree.h"
#include "TFile.h"
#include "TRandom.h"
#include "TError.h"

void parallelMergeTest(UInt_t nhist, UInt_t ndims = 1, UInt_t nbins = 100)
{

   gBenchmark->Start("parallelMergeTest");

   TFile *file = TFile::Open("mergedClient.root?pmerge=localhost:1095","RECREATE");

   Float_t px, py;
   TTree *tree = 0;
   switch (ndims) {
      case 1: {
         for(UInt_t h = 0 ; h < nhist; ++h) {
            new TH1F(TString::Format("hpx%d",h),"This is the px distribution",nbins,-4,4);
         }
         break;
      }
      case 2: {
         for(UInt_t h = 0 ; h < nhist; ++h) {
            new TH2F(TString::Format("hpxy%d",h),"py vs px",nbins,-4,4,nbins,-4,-4);
         }
         break;
      }
      case 99: {
         tree = new TTree("tree","tree");
         tree->SetAutoFlush(4000000);
         tree->Branch("px",&px);
         tree->Branch("py",&py);
      }
   }

   // Fill histogram randomly
   gRandom->SetSeed();
   const int kUPDATE = 1000000;
   for (int i = 0; i < 25000000; ) {
//      gRandom->Rannor(px,py);
//      if (idx%2 == 0)
//         hpx->Fill(px);
//      else
//         hpx->Fill(px,py);
      if(tree) tree->Fill();
      ++i;
      if (i && (i%kUPDATE) == 0) {
         file->Write();
      }
   }
   file->Write();
   delete file;

   gBenchmark->Show("parallelMergeTest");
}
