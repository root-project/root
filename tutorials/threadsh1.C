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

void *handle(void *ptr)
{
   int nr = (int) ptr;

   char name[5];
   sprintf(name,"hpx%d",nr);
   TThread::Lock();
   hpx[nr] = new TH1F(name,"This is the px distribution",100,-4,4);
   hpx[nr]->SetFillColor(48);
   TThread::UnLock();
   Float_t px, py, pz;
   gRandom->SetSeed();
   for (Int_t i=0;i<25000;i++) {
      c[nr]->cd();
      gRandom->Rannor(px,py);
      pz = px*px + py*py;
      Float_t random = gRandom->Rndm(1);
      hpx[nr]->Fill(px);
      if (i && (i%1000) == 0) {
         TThread::Printf("Here I am loop index: %3d , thread: %d", i, nr);
         if (i == 1000) hpx[nr]->Draw();
         c[nr]->Modified();
         c[nr]->Update();
      }
      gSystem->Sleep(1);
   }
   return 0;
}

void threadsh1()
{
#ifdef __CINT__
   printf("This script can only be executed via ACliC: .x threadsh1.C++\n");
   return;
#endif

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
   TThread::Ps();
}
