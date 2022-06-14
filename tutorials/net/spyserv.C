/// \file
/// \ingroup tutorial_net
/// Server program which allows clients, "spies", to connect and snoop objects.
/// To run this demo do the following:
///   - open two or more windows
///   - start root in all windows
///   - execute in the first window:    .x spyserv.C  (or spyserv.C++)
///   - execute in the other window(s): .x spy.C      (or spy.C++)
///   - in the "spy" client windows click the "Connect" button and snoop
///     the histograms by clicking on the "hpx", "hpxpy" and "hprof"
///     buttons
/// \macro_code
///
/// \author Fons Rademakers

#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "TSocket.h"
#include "TServerSocket.h"
#include "TMonitor.h"
#include "TMessage.h"
#include "TRandom.h"
#include "TList.h"
#include "TError.h"


class SpyServ {
private:
   TCanvas       *fCanvas;    // main canvas
   TH1F          *fHpx;       // 1-D histogram
   TH2F          *fHpxpy;     // 2-D histogram
   TProfile      *fHprof;     // profile histogram
   TServerSocket *fServ;      // server socket
   TMonitor      *fMon;       // socket monitor
   TList         *fSockets;   // list of open spy sockets
public:
   SpyServ();
   ~SpyServ();

   void HandleSocket(TSocket *s);
};


void SpyServ::HandleSocket(TSocket *s)
{
   if (s->IsA() == TServerSocket::Class()) {
      // accept new connection from spy
      TSocket *sock = ((TServerSocket*)s)->Accept();
      fMon->Add(sock);
      fSockets->Add(sock);
      printf("accepted connection from %s\n", sock->GetInetAddress().GetHostName());
   } else {
      // we only get string based requests from the spy
      char request[64];
      if (s->Recv(request, sizeof(request)) <= 0) {
         fMon->Remove(s);
         fSockets->Remove(s);
         printf("closed connection from %s\n", s->GetInetAddress().GetHostName());
         delete s;
         return;
      }

      // send requested object back
      TMessage answer(kMESS_OBJECT);
      if (!strcmp(request, "get hpx"))
         answer.WriteObject(fHpx);
      else if (!strcmp(request, "get hpxpy"))
         answer.WriteObject(fHpxpy);
      else if (!strcmp(request, "get hprof"))
         answer.WriteObject(fHprof);
      else
         Error("SpyServ::HandleSocket", "unexpected message");
      s->Send(answer);
   }
}

SpyServ::SpyServ()
{
   // Create the server process to fills a number of histograms.
   // A spy process can connect to it and ask for the histograms.
   // There is no apriori limit for the number of concurrent spy processes.

   // Open a server socket looking for connections on a named service or
   // on a specified port
   //TServerSocket *ss = new TServerSocket("spyserv", kTRUE);
   fServ = new TServerSocket(9090, kTRUE);
   if (!fServ->IsValid())
      gSystem->Exit(1);

   // Add server socket to monitor so we are notified when a client needs to be
   // accepted
   fMon  = new TMonitor;
   fMon->Add(fServ);

   // Create a list to contain all client connections
   fSockets = new TList;

   // Create a new canvas
   fCanvas = new TCanvas("SpyServ","SpyServ",200,10,700,500);
   fCanvas->SetFillColor(42);
   fCanvas->GetFrame()->SetFillColor(21);
   fCanvas->GetFrame()->SetBorderSize(6);
   fCanvas->GetFrame()->SetBorderMode(-1);

   // Create a 1-D, 2-D and a profile histogram
   fHpx    = new TH1F("hpx","This is the px distribution",100,-4,4);
   fHpxpy  = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
   fHprof  = new TProfile("hprof","Profile of pz versus px",100,-4,4,0,20);

   //  Set canvas/frame attributes (save old attributes)
   fHpx->SetFillColor(48);

   // Fill histograms randomly
   gRandom->SetSeed();
   Float_t px, py, pz;
   const Int_t kUPDATE = 1000;
   for (Int_t i = 0; ; i++) {
      gRandom->Rannor(px,py);
      pz = px*px + py*py;
      fHpx->Fill(px);
      fHpxpy->Fill(px,py);
      fHprof->Fill(px,pz);
      if (i && (i%kUPDATE) == 0) {
         if (i == kUPDATE) fHpx->Draw();
         fCanvas->Modified();
         fCanvas->Update();

         // Check if there is a message waiting on one of the sockets.
         // Wait not longer than 20ms (returns -1 in case of time-out).
         TSocket *s;
         if ((s = fMon->Select(20)) != (TSocket*)-1)
            HandleSocket(s);
         if (!fCanvas->TestBit(TObject::kNotDeleted))
            break;
         if (gROOT->IsInterrupted())
            break;
      }
   }
}

SpyServ::~SpyServ()
{
   // Clean up

   fSockets->Delete();
   delete fSockets;
   delete fServ;
   delete fCanvas;
   delete fHpx;
   delete fHpxpy;
   delete fHprof;
}

void spyserv()
{
   new SpyServ;
}
