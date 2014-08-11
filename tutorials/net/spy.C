// Client program which allows the snooping of objects from a spyserv process.
// To run this demo do the following (see spyserv.C):
//   - open two or more windows
//   - start root in all windows
//   - execute in the first window:    .x spyserv.C  (or spyserv.C++)
//   - execute in the other window(s): .x spy.C      (or spy.C++)
//   - in the "spy" client windows click the "Connect" button and snoop
//     the histograms by clicking on the "hpx", "hpxpy" and "hprof"
//     buttons
//Author: Fons Rademakers

#include "TGButton.h"
#include "TRootEmbeddedCanvas.h"
#include "TGLayout.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TSocket.h"
#include "TMessage.h"
#include "RQ_OBJECT.h"


class Spy {

RQ_OBJECT("Spy")

private:
   TGMainFrame         *fMain;
   TRootEmbeddedCanvas *fCanvas;
   TGHorizontalFrame   *fHorz;
   TGHorizontalFrame   *fHorz2;
   TGLayoutHints       *fLbut;
   TGLayoutHints       *fLhorz;
   TGLayoutHints       *fLcan;
   TGButton            *fHpx;
   TGButton            *fHpxpy;
   TGButton            *fHprof;
   TGButton            *fConnect;
   TGButton            *fQuit;
   TSocket             *fSock;
   TH1                 *fHist;

public:
   Spy();
   ~Spy();

   void Connect();
   void DoButton();
};

void Spy::DoButton()
{
   // Ask for histogram...

   if (!fSock->IsValid())
      return;

   TGButton *btn = (TGButton *) gTQSender;
   switch (btn->WidgetId()) {
      case 1:
         fSock->Send("get hpx");
         break;
      case 2:
         fSock->Send("get hpxpy");
         break;
      case 3:
         fSock->Send("get hprof");
         break;
   }
   TMessage *mess;
   if (fSock->Recv(mess) <= 0) {
      Error("Spy::DoButton", "error receiving message");
      return;
   }

   if (fHist) delete fHist;
   if (mess->GetClass()->InheritsFrom(TH1::Class())) {
      fHist = (TH1*) mess->ReadObject(mess->GetClass());
      if (mess->GetClass()->InheritsFrom(TH2::Class()))
         fHist->Draw("cont");
      else
         fHist->Draw();
      fCanvas->GetCanvas()->Modified();
      fCanvas->GetCanvas()->Update();
   }

   delete mess;
}

void Spy::Connect()
{
   // Connect to SpyServ
   fSock = new TSocket("localhost", 9090);
   fConnect->SetState(kButtonDisabled);
   fHpx->SetState(kButtonUp);
   fHpxpy->SetState(kButtonUp);
   fHprof->SetState(kButtonUp);
}

Spy::Spy()
{
   // Create a main frame
   fMain = new TGMainFrame(0, 100, 100);
   fMain->SetCleanup(kDeepCleanup);

   // Create an embedded canvas and add to the main frame, centered in x and y
   // and with 30 pixel margins all around
   fCanvas = new TRootEmbeddedCanvas("Canvas", fMain, 600, 400);
   fLcan = new TGLayoutHints(kLHintsCenterX|kLHintsCenterY,30,30,30,30);
   fMain->AddFrame(fCanvas, fLcan);

   // Create a horizonal frame containing three text buttons
   fLhorz = new TGLayoutHints(kLHintsExpandX, 0, 0, 0, 30);
   fHorz = new TGHorizontalFrame(fMain, 100, 100);
   fMain->AddFrame(fHorz, fLhorz);

   // Create three text buttons to get objects from server
   // Add to horizontal frame
   fLbut = new TGLayoutHints(kLHintsCenterX, 10, 10, 0, 0);
   fHpx = new TGTextButton(fHorz, "Get hpx", 1);
   fHpx->SetState(kButtonDisabled);
   fHpx->Connect("Clicked()", "Spy", this, "DoButton()");
   fHorz->AddFrame(fHpx, fLbut);
   fHpxpy = new TGTextButton(fHorz, "Get hpxpy", 2);
   fHpxpy->SetState(kButtonDisabled);
   fHpxpy->Connect("Clicked()", "Spy", this, "DoButton()");
   fHorz->AddFrame(fHpxpy, fLbut);
   fHprof = new TGTextButton(fHorz, "Get hprof", 3);
   fHprof->SetState(kButtonDisabled);
   fHprof->Connect("Clicked()", "Spy", this, "DoButton()");
   fHorz->AddFrame(fHprof, fLbut);

   // Create a horizonal frame containing two text buttons
   fHorz2 = new TGHorizontalFrame(fMain, 100, 100);
   fMain->AddFrame(fHorz2, fLhorz);

   // Create "Connect" and "Quit" buttons
   // Add to horizontal frame
   fConnect = new TGTextButton(fHorz2, "Connect");
   fConnect->Connect("Clicked()", "Spy", this, "Connect()");
   fHorz2->AddFrame(fConnect, fLbut);
   fQuit = new TGTextButton(fHorz2, "Quit");
   fQuit->SetCommand("gApplication->Terminate()");
   fHorz2->AddFrame(fQuit, fLbut);

   // Set main frame name, map sub windows (buttons), initialize layout
   // algorithm via Resize() and map main frame
   fMain->SetWindowName("Spy on SpyServ");
   fMain->MapSubwindows();
   fMain->Resize(fMain->GetDefaultSize());
   fMain->MapWindow();

   fHist = 0;
}

Spy::~Spy()
{
   // Clean up

   delete fHist;
   delete fSock;
   delete fLbut;
   delete fLhorz;
   delete fLcan;
   delete fHpx;
   delete fHpxpy;
   delete fHprof;
   delete fConnect;
   delete fQuit;
   delete fHorz;
   delete fHorz2;
   delete fCanvas;
   delete fMain;
}

void spy()
{
   new Spy;
}
