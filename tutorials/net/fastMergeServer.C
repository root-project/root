#include "TMessage.h"
#include "TBenchmark.h"
#include "TSocket.h"
#include "TH2.h"
#include "TTree.h"
#include "TMemFile.h"
#include "TRandom.h"
#include "TFileMerger.h"

#include "TServerSocket.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TMonitor.h"

#include "TFileCacheWrite.h"

char scratch[32*1024*1024];

void fastMergeServer(bool cache = false) {
   // This script shows how to make a simple iterative server that
   // can accept connections while handling currently open connections.
   // Compare this script to hserv.C that blocks on accept.
   // In this script a server socket is created and added to a monitor.
   // A monitor object is used to monitor connection requests on
   // the server socket. After accepting the connection
   // the new socket is added to the monitor and immediately ready
   // for use. Once two connections are accepted the server socket
   // is removed from the monitor and closed. The monitor continues
   // monitoring the sockets.
   //
   // To run this demo do the following:
   //   - Open three windows
   //   - Start ROOT in all three windows
   //   - Execute in the first window: .x hserv2.C
   //   - Execute in the second and third windows: .x hclient.C
   //Author: Fons Rademakers
   
   // Open a server socket looking for connections on a named service or
   // on a specified port.
   //TServerSocket *ss = new TServerSocket("rootserv", kTRUE);
   TServerSocket *ss = new TServerSocket(9090, kTRUE);
   if (!ss->IsValid()) {
      return;
   }

   TMonitor *mon = new TMonitor;
   
   mon->Add(ss);

   UInt_t clientCount = 0;
   TMemFile *transient = 0;
   
   TFileMerger merger(kFALSE,kFALSE);
   merger.OutputFile("hserv-output1.root",kTRUE);   
   if (cache) new TFileCacheWrite(merger.GetOutputFile(),32*1024*1024);
   while (1) {
      TMessage *mess;
      TSocket  *s;

      s = mon->Select();

      if (s->IsA() == TServerSocket::Class()) {
         if (clientCount > 100) {
            printf("only accept 100 clients connections\n");
            mon->Remove(ss);
            ss->Close();         
         } else {
            TSocket *client = ((TServerSocket *)s)->Accept();
            client->Send(TString::Format("go %d",clientCount));
            ++clientCount;
            mon->Add(client);
            printf("Accept %d connections\n",clientCount);
         }
         continue;
      }
      
      s->Recv(mess);

      if (mess->What() == kMESS_STRING) {
         char str[64];
         mess->ReadString(str, 64);
         printf("Client %d: %s\n", clientCount, str);
         mon->Remove(s);
         printf("Client %d: bytes recv = %d, bytes sent = %d\n", clientCount, s->GetBytesRecv(),
                s->GetBytesSent());
         s->Close();
         --clientCount;
         if (mon->GetActive() == 0 || clientCount == 0) {
            printf("No more active clients... stopping\n");
            break;
         }
      } else if (mess->What() == kMESS_ANY) {
         Long64_t length;
         mess->ReadLong64(length); // '*mess >> length;' is broken in CINT for Long64_t.
         // fprintf(stderr,"Seeing %lld bytes\n",length);

         delete transient;
         mess->ReadFastArray(scratch,length);
         transient = new TMemFile("hsimple.memroot",scratch,length);
//         mess->ReadFastArray(transient->GetBuffer(),end);
//         transient->SetOffset(0);
//         transient->SetEND(end);
//         transient->ReadKeys();

         merger.AddAdoptFile(transient);

         TH1 *h; transient->GetObject("hpx",h);
         if (h==0) transient->GetObject("hpxpy",h);
         TTree *in; transient->GetObject("tree",in);

         if (h) {
            h->Print();
         }
         if (in) {
            printf("Received %s with %lld\n",in->GetName(),in->GetEntries());
         }
         merger.IncrementalMerge();
         transient = 0;
      } else if (mess->What() == kMESS_OBJECT) {
         printf("got object of class: %s\n", mess->GetClass()->GetName());
      } else {
         printf("*** Unexpected message ***\n");
      }

      delete mess;
   }
}
