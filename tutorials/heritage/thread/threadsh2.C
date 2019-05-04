/// \file
/// \ingroup tutorial_thread
/// Example of a simple script creating 2 threads each with one canvas.
/// This script can only be executed via ACliC .x threadsh2.C++.
/// The canvases are saved in a animated gif file.
///
/// \macro_code
///
/// \author Victor Perevovchikov

#include "TROOT.h"
#include "TCanvas.h"
#include "TRootCanvas.h"
#include "TFrame.h"
#include "TH1F.h"
#include "TRandom.h"
#include "TThread.h"
#include "TMethodCall.h"

TCanvas *c1, *c2;
TH1F    *hpx, *total, *hmain, *s1, *s2;
TThread *thread1, *thread2, *threadj;
Bool_t   finished;

void *handle1(void *)
{
   int nfills = 10000;
   int upd = 500;

   TThread::Lock();
   hpx = new TH1F("hpx", "This is the px distribution", 100, -4, 4);
   hpx->SetFillColor(48);
   TThread::UnLock();
   Float_t px, py, pz;
   gRandom->SetSeed();
   for (Int_t i = 0; i < nfills; i++) {
      gRandom->Rannor(px, py);
      pz = px*px + py*py;
      hpx->Fill(px);
      if (i && (i%upd) == 0) {
         if (i == upd) {
            c1->cd();
            hpx->Draw();
         }
         c1->Modified();
         c1->Update();
         gSystem->Sleep(10);
         TMethodCall c(c1->IsA(), "Print", "");
         void *arr[4];
         arr[1] = &c;
         arr[2] = (void *)c1;
         arr[3] = (void*)"\"hpxanim.gif+50\"";
         (*gThreadXAR)("METH", 4, arr, NULL);
      }
   }
   c1->Modified();
   c1->Update();
   TMethodCall c(c1->IsA(), "Print", "");
   void *arr[4];
   arr[1] = &c;
   arr[2] = (void *)c1;
   arr[3] = (void*)"\"hpxanim.gif++\"";
   (*gThreadXAR)("METH", 4, arr, NULL);
   return 0;
}

void *handle2(void *)
{
   int nfills = 10000;
   int upd = 500;

   TThread::Lock();
   total  = new TH1F("total","This is the total distribution",100,-4,4);
   hmain  = new TH1F("hmain","Main contributor",100,-4,4);
   s1     = new TH1F("s1","This is the first signal",100,-4,4);
   s2     = new TH1F("s2","This is the second signal",100,-4,4);
   total->Sumw2();   // this makes sure that the sum of squares of weights will be stored
   total->SetMarkerStyle(21);
   total->SetMarkerSize(0.7);
   hmain->SetFillColor(16);
   s1->SetFillColor(42);
   s2->SetFillColor(46);
   TThread::UnLock();
   Float_t xs1, xs2, xmain;
   gRandom->SetSeed();
   for (Int_t i = 0; i < nfills; i++) {
      xmain = gRandom->Gaus(-1,1.5);
      xs1   = gRandom->Gaus(-0.5,0.5);
      xs2   = gRandom->Landau(1,0.15);
      hmain->Fill(xmain);
      s1->Fill(xs1,0.3);
      s2->Fill(xs2,0.2);
      total->Fill(xmain);
      total->Fill(xs1,0.3);
      total->Fill(xs2,0.2);
      if (i && (i%upd) == 0) {
         if (i == upd) {
            c2->cd();
            total->Draw("e1p");
            hmain->Draw("same");
            s1->Draw("same");
            s2->Draw("same");
         }
         c2->Modified();
         c2->Update();
         gSystem->Sleep(10);
         TMethodCall c(c2->IsA(), "Print", "");
         void *arr[4];
         arr[1] = &c;
         arr[2] = (void *)c2;
         arr[3] = (void*)"\"hsumanim.gif+50\"";
         (*gThreadXAR)("METH", 4, arr, NULL);
      }
   }
   total->Draw("sameaxis"); // to redraw axis hidden by the fill area
   c2->Modified();
   c2->Update();
   // make infinite animation by adding "++" to the file name
   TMethodCall c(c2->IsA(), "Print", "");
   void *arr[4];
   arr[1] = &c;
   arr[2] = (void *)c2;
   arr[3] = (void*)"\"hsumanim.gif++\"";
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

void tryclosing(Int_t id)
{
   // allow to close the canvas only after the threads are done
   if (!finished) return;
   if (id == 1) ((TRootCanvas *)c1->GetCanvasImp())->CloseWindow();
   else if (id == 2) ((TRootCanvas *)c2->GetCanvasImp())->CloseWindow();
}

#include "TClass.h"

void threadsh2()
{
   if (gROOT->IsBatch()) {
      return;
   }
   c1 = new TCanvas("c1","Dynamic Filling Example", 100, 30, 400, 300);
   c1->SetFillColor(42);
   c1->GetFrame()->SetFillColor(21);
   c1->GetFrame()->SetBorderSize(6);
   c1->GetFrame()->SetBorderMode(-1);
   // connect to the CloseWindow() signal and prevent to close the canvas
   // while the thread is running
   TRootCanvas *rc = dynamic_cast<TRootCanvas*>(c1->GetCanvasImp());
   if (!rc) return;

   rc->Connect("CloseWindow()", 0, 0,
               "tryclosing(Int_t=1)");
   rc->DontCallClose();

   c2 = new TCanvas("c2","Dynamic Filling Example", 515, 30, 400, 300);
   c2->SetGrid();
   // connect to the CloseWindow() signal and prevent to close the canvas
   // while the thread is running
   rc = dynamic_cast<TRootCanvas*>(c2->GetCanvasImp());
   if (!rc) return;
   rc->Connect("CloseWindow()", 0, 0,
               "tryclosing(Int_t=2)");
   rc->DontCallClose();

   finished = kFALSE;
   //gDebug = 1;
   gSystem->Unlink("hpxanim.gif");
   gSystem->Unlink("hsumanim.gif");

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
