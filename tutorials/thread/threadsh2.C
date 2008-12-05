// Example of a simple script creating 2 threads each with one canvas.
// The canvases are saved in a gif file.
// This script can only be executed via ACliC .x threadsh2.C++.

#include "TROOT.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "TH1F.h"
#include "TRandom.h"
#include "TThread.h"
#include "TMethodCall.h"

TH1F    *hpx[2];
TThread *thread1, *thread2, *threadj;
Bool_t finished;

void *handle1(void *ptr)
{
   long nr = (long) ptr;
   int nfills = 250000;
   int upd = 5000;

   TCanvas *c0 = new TCanvas("c0","Dynamic Filling Example", 100, 20, 400, 300);
   c0->SetFillColor(42);
   c0->GetFrame()->SetFillColor(21);
   c0->GetFrame()->SetBorderSize(6);
   c0->GetFrame()->SetBorderMode(-1);

   char name[32];
   sprintf(name,"hpx%ld",nr);
   TThread::Lock();
   hpx[nr] = new TH1F(name,"This is the px distribution",100,-4,4);
   hpx[nr]->SetFillColor(48);
   TThread::UnLock();
   Float_t px, py, pz;
   gRandom->SetSeed();
   for (Int_t i = 0; i < nfills; i++) {
      gRandom->Rannor(px,py);
      pz = px*px + py*py;
      hpx[nr]->Fill(px);
      if (i && (i%upd) == 0) {
         if (i == upd) {
            c0->cd();
            hpx[nr]->Draw();
         }
         c0->Modified();
         c0->Update();
         gSystem->Sleep(10);
      }
   }
   TMethodCall c(c0->IsA(), "Print", "");
   void *arr[4];
   arr[1] = &c;
   arr[2] = (void *)c0;
   arr[3] = (void*)"\"c0.gif\"";
   (*gThreadXAR)("METH", 4, arr, NULL);
   return 0;
}

void *handle2(void *ptr)
{
   long nr = (long) ptr;
   int nfills = 250000;
   int upd = 5000;

   TCanvas *c1 = new TCanvas("c1","Dynamic Filling Example", 100, 350, 400, 300);
   c1->SetFillColor(42);
   c1->GetFrame()->SetFillColor(21);
   c1->GetFrame()->SetBorderSize(6);
   c1->GetFrame()->SetBorderMode(-1);

   char name[32];
   sprintf(name,"hpx%ld",nr);
   TThread::Lock();
   hpx[nr] = new TH1F(name,"This is the px distribution",100,-4,4);
   hpx[nr]->SetFillColor(48);
   TThread::UnLock();
   Float_t px, py, pz;
   gRandom->SetSeed();
   for (Int_t i = 0; i < nfills; i++) {
      gRandom->Rannor(px,py);
      pz = px*px + py*py;
      hpx[nr]->Fill(px);
      if (i && (i%upd) == 0) {
         if (i == upd) {
            c1->cd();
            hpx[nr]->Draw();
         }
         c1->Modified();
         c1->Update();
         gSystem->Sleep(10);
      }
   }
   TMethodCall c(c1->IsA(), "Print", "");
   void *arr[4];
   arr[1] = &c;
   arr[2] = (void *)c1;
   arr[3] = (void*)"\"c1.gif\"";
   (*gThreadXAR)("METH", 4, arr, NULL);
   return 0;
}

void *joiner(void *)
{
   thread1->Join();
   thread2->Join();

   finished = kTRUE;

   return 0;
}

void threadsh2()
{
#ifdef __CINT__
   printf("This script can only be executed via ACliC: .x threadsh2.C++\n");
   return;
#endif

   finished = kFALSE;
   //gDebug = 1;

   printf("Starting Thread 0\n");
   thread1 = new TThread("t0", handle1, (void*) 0);
   thread1->Run();
   printf("Starting Thread 1\n");
   thread2 = new TThread("t1", handle2, (void*) 1);
   thread2->Run();
   printf("Starting Joiner Thread \n");
   threadj = new TThread("t4", joiner, (void*) 3);
   threadj->Run();

   TThread::Ps();

   while (!finished) {
      gSystem->Sleep(100);
      gSystem->ProcessEvents();
   }

   threadj->Join();
   TThread::Ps();

   delete thread1;
   delete thread2;
   delete threadj;
}
