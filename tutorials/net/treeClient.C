#include "TMessage.h"
#include "TBenchmark.h"
#include "TSocket.h"
#include "TH2.h"
#include "TTree.h"
#include "TMemFile.h"
#include "TRandom.h"

void treeClient(Bool_t evol=kFALSE) 
{
   // Client program which creates and fills 2 histograms and a TTree. 
   // Every 1000000 fills the histograms and TTree is send to the server which displays the histogram.
   //
   // To run this demo do the following:
   //   - Open at least 2 windows
   //   - Start ROOT in the first windows
   //   - Execute in the first window: .x fastMergeServer.C
   //   - Execute in the other windows: root.exe -b -l -q .x treeClient.C
   //     (You can put it in the background if wanted).
   // If you want to run the hserv.C on a different host, just change
   // "localhost" in the TSocket ctor below to the desired hostname.
   //
   //Author: Fons Rademakers, Philippe Canal
   
   gBenchmark->Start("treeClient");

   // Open connection to server
   TSocket *sock = new TSocket("localhost", 9090);
   if (!sock->IsValid()) {
      return;
   }

   // Wait till we get the start message
   char str[32];
   sock->Recv(str, 32);

   // server tells us who we are
   int idx = !strcmp(str, "go 0") ? 0 : 1;

   Float_t messlen  = 0;
   Float_t cmesslen = 0;

   TMemFile *file = new TMemFile("hsimple.memroot","RECREATE");
   TH1 *hpx;
   if (idx == 0) {
      // Create the histogram
      hpx = new TH1F("hpx","This is the px distribution",100,-4,4);
      hpx->SetFillColor(48);  // set nice fillcolor
   } else {
      hpx = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
   }
   Float_t px, py;
   TTree *tree = new TTree("tree","tree");
   tree->SetAutoFlush(4000000);
   tree->Branch("px",&px);
   tree->Branch("py",&py);
 
   TMessage::EnableSchemaEvolutionForAll(evol);
   TMessage mess(kMESS_OBJECT);

   // Fill histogram randomly
   gRandom->SetSeed();
   const int kUPDATE = 1000000;
   for (int i = 0; i < 25000000; ) {
      gRandom->Rannor(px,py);
      if (idx == 0)
         hpx->Fill(px);
      else
         hpx->Fill(px,py);
      tree->Fill();
      ++i;
      if (i && (i%kUPDATE) == 0) {
         file->Write();
         mess.Reset(kMESS_ANY);              // re-use TMessage object
         mess.WriteLong64(file->GetEND());   // 'mess << file->GetEND();' is broken in CINT for Long64_t
         mess.WriteFastArray(file->GetBuffer(),file->GetEND());
         sock->Send(mess);          // send message
         messlen  += mess.Length();
         cmesslen += mess.CompLength();
         
         file->ResetAfterMerge(0);
      }
   }
   sock->Send("Finished");          // tell server we are finished

   if (cmesslen > 0)
      printf("Average compression ratio: %g\n", messlen/cmesslen);

   gBenchmark->Show("hclient");

   // Close the socket
   sock->Close();
}
