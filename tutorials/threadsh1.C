// Example of a simple script creating 3 threads.
// This script can only be executed via ACliC .x threadsh1.C++.
// Before executing the script, load the Thread library with:
//  gSystem->Load("libThread");
// This is not needed anymore due to the rootmap facility which
// automatically loads the needed libraries.

#include "TCanvas.h"
#include "TFrame.h"
#include "TPad.h"
#include "TH1F.h"
#include "TRandom.h"
#include "TThread.h"
#include <Riostream.h>

TCanvas *c[5];
TH1F *hpx[5];
Bool_t finished;

void *handle(void *ptr)
{
   long nr = (long) ptr;
   int nfills = 2500000;
   int upd = 5000;

   char name[5];
   sprintf(name,"hpx%ld",nr);
   TThread::Lock();
   hpx[nr] = new TH1F(name,"This is the px distribution",100,-4,4);
   hpx[nr]->SetFillColor(48);
   TThread::UnLock();
   Float_t px, py, pz;
   gRandom->SetSeed();
   for (Int_t i=0;i<nfills;i++) {
      c[nr]->cd();
      gRandom->Rannor(px,py);
      pz = px*px + py*py;
      Float_t random = gRandom->Rndm(1);
      hpx[nr]->Fill(px);
      if (i && (i%upd) == 0) {
         //TThread::Printf("Here I am loop index: %3d , thread: %ld", i, nr);
         //printf("Here I am loop index: %3d , thread: %ld\n", i, nr);
         if (i == upd) hpx[nr]->Draw();
         c[nr]->Modified();
         //c[nr]->Update();
      }
      //TThread::Sleep(0, 10000000);
   }
   return 0;
}

void *updater(void *ptr)
{
   long nthrds = (long) ptr;
   int i;

   while (!finished) {
      for (i = 1; i <= nthrds; i++) {
         if (c[i]->IsModified()) {
            printf("Update canvas %d\n", i);
            c[i]->Update();
         }
      }
      //TThread::Sleep(0, 10000000);
      //gSystem->Sleep(10);
   }
   for (i = 1; i <= nthrds; i++) {
      c[i]->Modified();
      c[i]->Update();
   }
   return 0;
}

void threadsh1()
{
#ifdef __CINT__
   printf("This script can only be executed via ACliC: .x threadsh1.C++\n");
   return;
#endif

   finished = kFALSE;
   gDebug = 1;

   c[1] = new TCanvas("c1","Dynamic Filling Example",100,20,400,300);
   c[1]->SetFillColor(42);
   c[1]->GetFrame()->SetFillColor(21);
   c[1]->GetFrame()->SetBorderSize(6);
   c[1]->GetFrame()->SetBorderMode(-1);
   c[2] = new TCanvas("c2","Dynamic Filling Example",510,20,400,300);
   c[2]->SetFillColor(42);
   c[2]->GetFrame()->SetFillColor(21);
   c[2]->GetFrame()->SetBorderSize(6);
   c[2]->GetFrame()->SetBorderMode(-1);
   c[3] = new TCanvas("c3","Dynamic Filling Example",100,350,400,300);
   c[3]->SetFillColor(42);
   c[3]->GetFrame()->SetFillColor(21);
   c[3]->GetFrame()->SetBorderSize(6);
   c[3]->GetFrame()->SetBorderMode(-1);

   printf("Starting Thread 0\n");
   TThread *h0 = new TThread("h0", updater, (void*) 3);
   h0->Run();
   printf("Starting Thread 1\n");
   TThread *h1 = new TThread("h1", handle, (void*) 1);
   h1->Run();
   printf("Starting Thread 2\n");
   TThread *h2 = new TThread("h2", handle, (void*) 2);
   h2->Run();
   printf("Starting Thread 3\n");
   TThread *h3 = new TThread("h3", handle, (void*) 3);
   h3->Run();

   TThread::Ps();

   h1->Join();
   TThread::Ps();
   h2->Join();
   h3->Join();
   finished = kTRUE;
   h0->Join();
   TThread::Ps();
}
