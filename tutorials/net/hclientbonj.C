#include "TBenchmark.h"
#include "TList.h"
#include "TInetAddress.h"
#include "TSocket.h"
#include "TMessage.h"
#include "TH1.h"
#include "TH2.h"
#include "TRandom.h"
#include "TBonjourBrowser.h"
#include "TBonjourResolver.h"
#include "TBonjourRecord.h"


static Bool_t gEvo = kFALSE;

void ConnectToServer(const TInetAddress *hostb, Int_t port)
{
   // Called by the Bonjour resolver with the host and port to which
   // we can connect.

   // Connect only once...
   TBonjourResolver *resolver = (TBonjourResolver*) gTQSender;
   TInetAddress host = *hostb;
   delete resolver;

   printf("ConnectToServer: host = %s, port = %d\n", host.GetHostName(), port);

   //--- Here starts original hclient.C code ---

   // Open connection to server
   TSocket *sock = new TSocket(host.GetHostName(), port);

   // Wait till we get the start message
   char str[32];
   sock->Recv(str, 32);

   // server tells us who we are
   int idx = !strcmp(str, "go 0") ? 0 : 1;

   Float_t messlen  = 0;
   Float_t cmesslen = 0;
   if (idx == 1)
      sock->SetCompressionLevel(1);

   TH1 *hpx;
   if (idx == 0) {
      // Create the histogram
      hpx = new TH1F("hpx","This is the px distribution",100,-4,4);
      hpx->SetFillColor(48);  // set nice fillcolor
   } else {
      hpx = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
   }

   TMessage::EnableSchemaEvolutionForAll(gEvo);
   TMessage mess(kMESS_OBJECT);
   //TMessage mess(kMESS_OBJECT | kMESS_ACK);

   // Fill histogram randomly
   gRandom->SetSeed();
   Float_t px, py;
   const int kUPDATE = 1000;
   for (int i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      if (idx == 0)
         hpx->Fill(px);
      else
         hpx->Fill(px,py);
      if (i && (i%kUPDATE) == 0) {
         mess.Reset();              // re-use TMessage object
         mess.WriteObject(hpx);     // write object in message buffer
         sock->Send(mess);          // send message
         messlen  += mess.Length();
         cmesslen += mess.CompLength();
      }
   }
   sock->Send("Finished");          // tell server we are finished

   if (cmesslen > 0)
      printf("Average compression ratio: %g\n", messlen/cmesslen);

   gBenchmark->Show("hclient");

   // Close the socket
   sock->Close();
}

void UpdateBonjourRecords(TList *records)
{
   // Browse for Bonjour record of type "_hserv2._tcp." in domain "local.".
   // When found, create Bonjour resolver to get host and port of this record.

   static Bool_t resolved = kFALSE;

   // we can be called multiple times whenever a new server appears
   printf("UpdateBonjourRecords (resolved = %s)\n", resolved ? "kTRUE" : "kFALSE");

   if (resolved) return;

   // Look for _hserv2._tcp. in local. domain and try to resolve it
   TBonjourRecord *rec;
   TIter next(records);
   while ((rec = (TBonjourRecord*) next())) {
      if (!strcmp(rec->GetRegisteredType(), "_hserv2._tcp.") &&
          !strcmp(rec->GetReplyDomain(), "local.")) {
         rec->Print();
         TBonjourResolver *resolver = new TBonjourResolver;
         resolver->Connect("RecordResolved(TInetAddress*,Int_t)", 0, 0,
                           "ConnectToServer(TInetAddress*,Int_t)");
         resolver->ResolveBonjourRecord(*rec);
         resolved = kTRUE;
      }
   }
}

void hclientbonj(Bool_t evol=kFALSE)
{
   // Client program which creates and fills a histogram. Every 1000 fills
   // the histogram is send to the server which displays the histogram.
   //
   // To run this demo do the following:
   //   - Open three windows
   //   - Start ROOT in all three windows
   //   - Execute in the first window: .x hserv.C (or hserv2.C)
   //   - Execute in the second and third windows: .x hclient.C
   // If you want to run the hserv.C on a different host, just change
   // "localhost" in the TSocket ctor below to the desired hostname.
   //
   // The script argument "evol" can be used when using a modified version
   // of the script where the clients and server are on systems with
   // different versions of ROOT. When evol is set to kTRUE the socket will
   // support automatic schema evolution between the client and the server.
   //
   //Author: Fons Rademakers

   gEvo = evol;

   gBenchmark->Start("hclient");

   TBonjourBrowser *browser = new TBonjourBrowser;
   browser->Connect("CurrentBonjourRecordsChanged(TList*)", 0, 0,
                    "UpdateBonjourRecords(TList*)");
   browser->BrowseForServiceType("_hserv2._tcp");
}
